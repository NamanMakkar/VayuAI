# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from vajra.ops import xywh2xyxy, xyxy_to_xywh, crop_mask
from vajra.tal.tal_assigner import TaskAlignedAssigner, RotatedTaskAlignedAssigner
from vajra.tal.assigner_utils import dist_calculator
from vajra.tal.anchor_generator import generate_anchors, dist2bbox, bbox2dist, dist2rbox
from vajra.metrics import bbox_iou, probabilistic_iou, OKS_SIGMA
from vajra.utils import LOGGER
from vajra.utils.torch_utils import autocast

class VarifocalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') * weight).mean(1).sum()
        return loss

class DFLoss(nn.Module):
    def __init__(self, reg_max=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(pred, label, gamma=1.5, alpha=0.25):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class DetectionLoss:
    def __init__(self, model, tal_topk=10) -> None:
        device = next(model.parameters()).device
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        head = model.model[-1]
        self.use_dfl = head.reg_max > 1
        self.reg_max = head.reg_max if head else 16
        self.hyperparms = model.args
        self.stride = model.stride
        self.num_classes = model.num_classes
        self.num_outputs = self.num_classes + self.reg_max * 4
        self.device = device
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.num_classes, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        num_layers, num_extra = targets.shape

        if num_layers == 0:
            out = torch.zeros(batch_size, 0, num_extra - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), num_extra - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch, anchors, channels = pred_dist.shape
            pred_dist = pred_dist.view(batch, anchors, 4, channels//4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.num_outputs, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.num_classes), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = generate_anchors(feats, self.stride, 0.5)
        # Targets

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        loss[0] *= self.hyperparms.box  # box gain
        loss[1] *= self.hyperparms.cls  # cls gain
        loss[2] *= self.hyperparms.dfl  # dfl gain
        return loss.sum() * batch_size, loss.detach()

class DEYODetectionLoss(DetectionLoss):
    def __init__(self, model, tal_topk=1) -> None:
        super().__init__(model, tal_topk)

    def __call__(self, preds, batch):
        loss = torch.zeros(1, device=self.device)
        feats, pred_scores, topk_ind = preds[1] if isinstance(preds[1], tuple) else preds

        pred_distri, enc_scores = torch.cat([xi.view(feats[0].shape[0], self.num_outputs, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.num_classes), 1
        )

        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        enc_scores = enc_scores.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[1]
        img_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = generate_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=img_size[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                enc_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[0] = self.bce(enc_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        
        loss[0] *= self.hyperparms.cls  # cls gain
        
        for pred_score in pred_scores:
            bs, nq = pred_score.shape[:2]
            batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, nq).view(-1)
            total_scores = torch.zeros((bs, pred_bboxes.shape[1], self.num_classes), dtype=pred_score.dtype, device=pred_score.device)
            total_scores[batch_ind, topk_ind] = pred_score.view(bs * nq, -1).sigmoid()
            
            
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                total_scores.detach(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            
            target_scores = target_scores[batch_ind, topk_ind].view(bs, nq, -1)
            target_scores_sum = max(target_scores.sum(), 1)
            
            # Cls loss
            
            loss_ = self.bce(pred_score, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
            loss_ *= self.hyperparms.cls  # cls gain   
            loss[0] += loss_
        return loss.sum() * batch_size, loss.detach()

class SegmentationLoss(DetectionLoss):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image"""
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, num_masks, mask_h, mask_w = proto.shape
        pred_distribution, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.num_outputs, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distribution = pred_distribution.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        img_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = generate_anchors(feats, self.stride, 0.5)

        try:
            batch_size = pred_scores.shape[0]
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=img_size[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR! - Segmentation dataset incorrectly formatted or not a segmentation dataset.\n'
                            "This error can occur when incorrectly training a 'segmentation' model on a 'detection' dataset")

        pred_bboxes = self.bbox_decode(anchor_points, pred_distribution)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(pred_distribution, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)

            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, img_size, self.overlap)

        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyperparms.box
        loss[1] *= self.hyperparms.box
        loss[2] *= self.hyperparms.cls
        loss[3] *= self.hyperparms.dfl

        return loss.sum() * batch_size, loss.detach()

    def calculate_segmentation_loss(self,
                                    fg_mask: torch.Tensor,
                                    masks: torch.Tensor,
                                    target_gt_idx: torch.Tensor,
                                    target_bboxes: torch.Tensor,
                                    batch_idx: torch.Tensor,
                                    proto: torch.Tensor,
                                    pred_masks: torch.Tensor,
                                    img_size: torch.Tensor,
                                    overlap: bool) -> torch.Tensor:

        _, _, mask_h, mask_w = proto.shape
        loss = 0

        target_bboxes_normalized = target_bboxes / img_size[[1, 0, 1, 0]]

        marea = xyxy_to_xywh(target_bboxes_normalized)[..., 2:].prod(2)

        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask=gt_mask, pred=pred_masks_i[fg_mask_i], proto=proto_i, xyxy=mxyxy_i[fg_mask_i], area=marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()

class PanopticLoss(SegmentationLoss):
    def __init__(self, model) -> None:
        super().__init__(model)

    def __call__(self, preds, batch):
        loss = torch.zeros(6, device=self.device)
        feats, pred_masks, proto, pse_masks = preds if len(preds) == 3 else preds[1]
        batch_size, num_masks, mask_h, mask_w = proto.shape
        pred_distribution, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.num_outputs, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.num_classes), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distribution = pred_distribution.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        img_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = generate_anchors(feats, self.stride, 0.5)

        try:
            batch_size = pred_scores.shape[0]
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=img_size[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR! - Segmentation dataset incorrectly formatted or not a segmentation dataset.\n'
                            "This error can occur when incorrectly training a 'segmentation' model on a 'detection' dataset")

        pred_bboxes = self.bbox_decode(anchor_points, pred_distribution)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(pred_distribution, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)

            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, img_size, self.overlap)

        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        se_masks = batch["segments"].to(self.device).float()
        pt = torch.flatten(pse_masks, start_dim=2).permute(0, 2, 1)
        gt = torch.flatten(se_masks, start_dim=2).permute(0, 2, 1)

        bs, _, _ = gt.shape

        total_loss = (sigmoid_focal_loss(pt, gt, alpha = .25, gamma = 2., reduction = 'mean')) / 2.
        loss[4] += total_loss * 20.
        pt = torch.flatten(pse_masks.softmax(dim = 1))
        gt = torch.flatten(se_masks)

        inter_mask = torch.sum(torch.mul(pt, gt))
        union_mask = torch.sum(torch.add(pt, gt))
        dice_coef = (2. * inter_mask + 1.) / (union_mask + 1.)
        loss[5] += (1. - dice_coef) / 2.

        loss[0] *= self.hyperparms.box
        loss[1] *= self.hyperparms.box
        loss[2] *= self.hyperparms.cls
        loss[3] *= self.hyperparms.dfl
        loss[4] *= 2.5
        loss[5] *= 2.5

        return loss.sum() * batch_size, loss.detach()

class ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='mean')
        loss_items = loss.detach()
        return loss, loss_items

class RotatedBboxLoss(BboxLoss):
    def __init__(self, reg_max, use_dfl=False) -> None:
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probabilistic_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

class OBBLoss(DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.num_classes, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max, use_dfl=self.use_dfl).to(self.device)
    
    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)

        return out

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.num_outputs, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.num_classes), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        img_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = generate_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * img_size[0].item(), targets[:, 5] * img_size[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=img_size[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR! OBB dataset incorrectly formatted or not an OBB dataset.\n"
                "This error can occur when incorrectly training an 'OBB' model on a 'detect' dataset, "
                "i.e 'vajra train model='vajra-v1-nano-obb' data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml'"
            ) from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()
        
        loss[0] *= self.hyperparms.box
        loss[1] *= self.hyperparms.cls
        loss[2] *= self.hyperparms.dfl


        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            batch, anchors, channels = pred_dist.shape
            pred_dist = pred_dist.view(batch, anchors, 4, channels // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

class KeypointLoss(nn.Module):
    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class PoseLoss(DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.kpt_shape = model.head.keypoint_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        num_keypoints = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(num_keypoints, device=self.device) / num_keypoints
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.num_outputs, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.num_classes), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        img_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = generate_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=img_size[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= img_size[1]
            keypoints[..., 1] *= img_size[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )
        
        loss[0] *= self.hyperparms.box
        loss[1] *= self.hyperparms.pose
        loss[2] *= self.hyperparms.kobj
        loss[3] *= self.hyperparms.cls
        loss[4] *= self.hyperparms.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_keypoints = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros(
            (batch_size, max_keypoints, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]]

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )
        
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        keypoints_loss = 0
        keypoints_obj_loss = 0
        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy_to_xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            keypoints_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                keypoints_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        
        return keypoints_loss, keypoints_obj_loss
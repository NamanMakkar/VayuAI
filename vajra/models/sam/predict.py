# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from vajra.dataset.augment import LetterBox
from vajra.core.predictor import Predictor
from vajra.core.results import Results
from vajra.utils import HYPERPARAMS_CFG
from vajra import ops
from vajra.utils.torch_utils import select_device
from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .build import build_sam

class SamPredictor(Predictor):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        if model_configuration is None:
            model_configuration = {}
        model_configuration.update(dict(task="segment", mode="predict"))
        super().__init__(config, model_configuration, _callbacks)
        self.args.retina_masks = True
        self.img = None
        self.features = None
        self.prompts={}
        self.segment_all = False

    def preprocess(self, img):
        if self.img is not None:
            return self.img
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        if not_tensor:
            img = (img - self.mean) / self.std
        return img

    def pre_transform(self, img):
        assert len(img) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.args.img_size, auto=False, center=False)
        return [letterbox(image=x) for x in img]

    def inference(self, img, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(img, *args, **kwargs)
        return self.prompt_inference(img, bboxes, points, labels, masks, multimask_output)
    
    def prompt_inference(self, img, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        features = self.get_img_features(img) if self.features is None else self.features

        bboxes, points, labels, masks = self._prepare_prompts(img.shape[2:], bboxes, points, labels, masks)
        points = (points, labels) if points is not None else None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points = points, boxes=bboxes, masks=masks)

        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings = features,
            image_pe = self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )

        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)
    
    def _prepare_prompts(self, dst_shape, bboxes=None, points=None, labels=None, masks=None):
        src_shape = self.batch[1][0].shape[:2]
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])

        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            points = points[None] if points.ndim == 1 else points

            if labels is None:
                labels = np.ones(points.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert(
                points.shape[-2] == labels.shape[-1]
            ), f"Number of points {points.shape[-2]} should match number of labels {labels.shape[-1]}."
            points *= r
            if points.ndim == 2:
                points, labels = points[:, None, :], labels[:, None]

        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r

        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)
        return bboxes, points, labels, masks

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        import torchvision
        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers, crop_overlap_ratio)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(points_stride, crop_n_layers, crop_downscale_factor)
        pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])  # w, h
            # Crop image and interpolate to input size
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode="bilinear", align_corners=False)
            # (num_points, 2)
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points,) in batch_iterator(points_batch_size, points_for_image):
                pred_mask, pred_score = self.prompt_inference(crop_im, points=points, multimask_output=True)
                # Interpolate predicted masks to input size
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode="bilinear", align_corners=False)[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(
                    pred_mask, self.model.mask_threshold, stability_score_offset
                )
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # Bool type is much more memory-efficient.
                pred_mask = pred_mask > self.model.mask_threshold
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 0, iw, ih])
                if not torch.all(keep_mask):
                    pred_bbox, pred_mask, pred_score = pred_bbox[keep_mask], pred_mask[keep_mask], pred_score[keep_mask]

                crop_masks.append(pred_mask)
                crop_bboxes.append(pred_bbox)
                crop_scores.append(pred_score)

            # Do nms within this crop
            crop_masks = torch.cat(crop_masks)
            crop_bboxes = torch.cat(crop_bboxes)
            crop_scores = torch.cat(crop_scores)
            keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # NMS
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
            crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
            crop_scores = crop_scores[keep]

            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)
            region_areas.append(area.expand(len(crop_masks)))

        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)
        region_areas = torch.cat(region_areas)

        # Remove duplicate masks between crops
        if len(crop_regions) > 1:
            scores = 1 / region_areas
            keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
            pred_masks, pred_bboxes, pred_scores = pred_masks[keep], pred_bboxes[keep], pred_scores[keep]

        return pred_masks, pred_scores, pred_bboxes
    
    def setup_model(self, model, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

    def get_model(self):
        return build_sam(self.args.model)
    
    def postprocess(self, preds, img, orig_imgs):
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if len(masks) == 0:
                masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # to bool
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # NOTE: SAM models do not return cls info. This `cls` here is just a placeholder for consistency.
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # Reset segment-all mode.
        self.segment_all = False
        return results
    
    def setup_source(self, source):
        if source is not None:
            super().setup_source(source)

    def set_image(self, image):
        if self.model is None:
            self.setup_model(model=None)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_img_features(im)
            break

    def get_img_features(self, img):
        assert (
            isinstance(self.img_size, (tuple, list)) and self.img_size[0] == self.img_size[1]
        ), f"SAM models only support square image size, but got {self.img_size}."
        self.model.set_img_size(self.img_size)
        return self.model.image_encoder(img)
    
    def set_prompts(self, prompts):
        self.prompts = prompts

    def reset_image(self):
        self.img = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        import torchvision

        if len(masks) == 0:
            return masks
        
        new_masks = []
        scores = []

        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(float(unchanged))

        new_masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(new_masks)
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)

        return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep
    
class SAM2Predictor(SamPredictor):
    """
    SAM2Predictor class for advanced image segmentation using Segment Anything Model 2 architecture.

    This class extends the base Predictor class to implement SAM2-specific functionality for image
    segmentation tasks. It provides methods for model initialization, feature extraction, and
    prompt-based inference.

    Attributes:
        _bb_feat_sizes (List[Tuple[int, int]]): Feature sizes for different backbone levels.
        model (torch.nn.Module): The loaded SAM2 model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        features (Dict[str, torch.Tensor]): Cached image features for efficient inference.
        segment_all (bool): Flag to indicate if all segments should be predicted.
        prompts (Dict): Dictionary to store various types of prompts for inference.

    Methods:
        get_model: Retrieves and initializes the SAM2 model.
        prompt_inference: Performs image segmentation inference based on various prompts.
        set_image: Preprocesses and sets a single image for inference.
        get_im_features: Extracts and processes image features using SAM2's image encoder.

    Examples:
        >>> predictor = SAM2Predictor(cfg)
        >>> predictor.set_image("path/to/image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> masks, scores, _ = predictor.prompt_inference(predictor.im, bboxes=bboxes)
        >>> print(f"Predicted {len(masks)} masks with average score {scores.mean():.2f}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]

    def get_model(self):
        """Retrieves and initializes the Segment Anything Model 2 (SAM2) for image segmentation tasks."""
        return build_sam(self.args.model)
    
    def prompt_inference(
        self,
        img,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        """
        Performs image segmentation inference based on various prompts using SAM2 architecture.

        This method leverages the Segment Anything Model 2 (SAM2) to generate segmentation masks for input images
        based on provided prompts such as bounding boxes, points, or existing masks. It supports both single and
        multi-object prediction scenarios.

        Args:
            im (torch.Tensor): Preprocessed input image tensor with shape (N, C, H, W).
            bboxes (np.ndarray | List[List[float]] | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | List[List[float]] | None): Object location points with shape (N, 2), in pixels.
            labels (np.ndarray | List[int] | None): Point prompt labels with shape (N,). 1 = foreground, 0 = background.
            masks (np.ndarray | None): Low-resolution masks from previous predictions with shape (N, H, W).
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.
            img_idx (int): Index of the image in the batch to process.

        Returns:
            (tuple): Tuple containing:
                - np.ndarray: Output masks with shape (C, H, W), where C is the number of generated masks.
                - np.ndarray: Quality scores for each mask, with length C.
                - np.ndarray: Low-resolution logits with shape (C, 256, 256) for subsequent inference.

        Examples:
            >>> predictor = SAM2Predictor(cfg)
            >>> image = torch.rand(1, 3, 640, 640)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores, logits = predictor.prompt_inference(image, bboxes=bboxes)
            >>> print(f"Generated {masks.shape[0]} masks with average score {scores.mean():.2f}")

        Notes:
            - The method supports batched inference for multiple objects when points or bboxes are provided.
            - Input prompts (bboxes, points) are automatically scaled to match the input image dimensions.
            - When both bboxes and points are provided, they are merged into a single 'points' input for the model.

        References:
            - SAM2 Paper: https://arxiv.org/pdf/2408.00714
        """
        features = self.get_img_features(img) if self.features is None else self.features

        bboxes, points, labels, masks = self._prepare_prompts(img.shape[2:], bboxes, points, labels, masks)
        points = (points, labels) if points is not None else None

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # Predict masks
        batched_mode = points is not None and points[0].shape[0] > 1  # multi object prediction
        high_res_features = None
        if isinstance(features, dict):
            high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
            features = features["image_embed"][[img_idx]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` could be 1 or 3 depends on `multimask_output`.
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)
    
    def _prepare_prompts(self, dst_shape, bboxes=None, points=None, labels=None, masks=None):
        """
        Prepares and transforms the input prompts for processing based on the destination shape.

        Args:
            dst_shape (tuple): The target shape (height, width) for the prompts.
            bboxes (np.ndarray | List | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | List | None): Points indicating object locations with shape (N, 2) or (N, num_points, 2), in pixels.
            labels (np.ndarray | List | None): Point prompt labels with shape (N,) or (N, num_points). 1 for foreground, 0 for background.
            masks (List | np.ndarray, Optional): Masks for the objects, where each mask is a 2D array.

        Raises:
            AssertionError: If the number of points don't match the number of labels, in case labels were passed.

        Returns:
            (tuple): A tuple containing transformed bounding boxes, points, labels, and masks.
        """
        bboxes, points, labels, masks = super()._prepare_prompts(dst_shape, bboxes, points, labels, masks)
        if bboxes is not None:
            bboxes = bboxes.view(-1, 2, 2)
            bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(len(bboxes), -1)
            # NOTE: merge "boxes" and "points" into a single "points" input
            # (where boxes are added at the beginning) to model.sam_prompt_encoder
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = bboxes, bbox_labels
        return bboxes, points, labels, masks
    
    def set_image(self, image):
        """
        Preprocesses and sets a single image for inference using the SAM2 model.

        This method initializes the model if not already done, configures the data source to the specified image,
        and preprocesses the image for feature extraction. It supports setting only one image at a time.

        Args:
            image (str | np.ndarray): Path to the image file as a string, or a numpy array representing the image.

        Raises:
            AssertionError: If more than one image is attempted to be set.

        Examples:
            >>> predictor = SAM2Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(np.array([...]))  # Using a numpy array

        Notes:
            - This method must be called before performing any inference on a new image.
            - The method caches the extracted features for efficient subsequent inferences on the same image.
            - Only one image can be set at a time. To process multiple images, call this method for each new image.
        """
        if self.model is None:
            self.setup_model(model=None)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            img = self.preprocess(batch[1])
            self.features = self.get_img_features(img)
            break

    def get_img_features(self, img):
        """Extracts image features from the SAM image encoder for subsequent processing."""
        assert (
            isinstance(self.img_size, (tuple, list)) and self.img_size[0] == self.img_size[1]
        ), f"SAM 2 models only support square image size, but got {self.img_size}."
        self.model.set_img_size(self.img_size)
        self._bb_feat_sizes = [[x // (4 * i) for x in self.img_size] for i in [1, 2, 4]]

        backbone_out = self.model.forward_image(img)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
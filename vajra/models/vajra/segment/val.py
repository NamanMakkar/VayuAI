# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torch.nn.functional as F
import numpy as np

from multiprocessing.pool import ThreadPool
from pathlib import Path
from vajra.checks import check_requirements
from vajra.models.vajra.detect import DetectionValidator
from vajra.utils import LOGGER, NUM_THREADS
from vajra import ops
from vajra.metrics import SegmentationMetrics, box_iou, mask_iou
from vajra.plotting import output_to_target, plot_images

class SegmentationValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentationMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        super().init_metrics(model)
        self.plot_masks = []

        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(tp_mask=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        return ( "%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.num_classes
        )
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
        return p, proto

    def _prepare_batch(self, si, batch):
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["img_size"])
        return predn, pred_masks

    def update_metrics(self, preds, batch):
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls = torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_mask=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            gt_masks = pbatch.pop("masks")
            if self.args.single_cls:
                pred[:, 5] = 0

            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_mask"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())

            if self.args.save_json:
                pred_masks = ops.scale_img(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    pbatch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.pred_to_json(predn, batch["im_file"][si], pred_masks)

            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:
            iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
        self.plot_masks.clear()

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        from vajra.core.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        from pycocotools.mask import encode

        def single_encode(x):
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy_to_xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        pred_masks = np.transpose(pred_masks, (2, 0, 1))

        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val_2017.json"
            pred_json = self.save_dir / "predictions.json"
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))
                pred = anno.loadRes(str(pred_json))
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[:2]
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
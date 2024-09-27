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
from vajra.models import vajra
from vajra.metrics import SegmentationMetrics, box_iou, mask_iou
from vajra.plotting import output_to_target, plot_images

class PanopticSegmentationValidator(vajra.segment.SegmentationValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "panoptic"
        self.metrics = SegmentationMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        batch["segments"] = batch["segments"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.num_classes
        )
        proto = preds[1][-2] if len(preds[1]) == 4 else preds[1]
        semask = preds[1][-1] if len(preds[1]) == 4 else preds[2]
        return p, proto, semask

    def _prepare_pred(self, pred, pbatch, proto):
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["img_size"])
        return predn, pred_masks
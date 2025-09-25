# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch

from vajra.dataset import VajraDetDataset
from vajra.dataset.augment import Compose, Format, vajra_transforms
from vajra.models.vajra.detect import DetectionValidator
from vajra.utils import colorstr
from vajra.configs import get_config, get_save_dir
from vajra.callbacks import get_default_callbacks, add_integration_callbacks
from vajra.checks import check_img_size
from vajra.utils import HYPERPARAMS_DETR_CFG_DICT
from vajra.metrics import DetectionMetrics
from vajra.ops import *

class DETRDataset(VajraDetDataset):
    def __init__(self, *args, data=None, task="detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def load_image(self, i, rect_mode=False):
        return super().load_image(i, rect_mode=rect_mode)
    
    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = vajra_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            transforms = Compose([])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms
    
class DETRValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        self.args = get_config(config=HYPERPARAMS_DETR_CFG_DICT, model_configuration=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.num_classes = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.img_size = check_img_size(self.args.img_size, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or get_default_callbacks()
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetectionMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.lb = []

    def build_dataset(self, img_path, mode="val", batch=None):
        return DETRDataset(
            img_path = img_path,
            imgsz = self.args.img_size,
            batch_size = batch,
            augment = False,
            hyp = self.args,
            rect=False,
            cache = self.args.cache or None,
            prefix = colorstr(f"{mode}: "),
            data = self.data
        )
    
    def postprocess(self, preds):
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.img_size
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):
            bbox = xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred
        return outputs
    
    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        img_size = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]

        if len(cls):
            bbox = xywh2xyxy(bbox)
            bbox[..., [0, 2]] *= ori_shape[1]
            bbox[..., [1, 3]] *= ori_shape[0]
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "img_size": img_size, "ratio_pad": ratio_pad}
    
    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.img_size
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.img_size
        return predn.float()
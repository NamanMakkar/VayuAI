# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import json
import torchvision
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
from vajra.utils import LOGGER, TQDM, colorstr
from vajra.nn.backend import Backend
from vajra.dataset.utils import check_cls_dataset, check_det_dataset
from vajra.utils.torch_utils import de_parallel, select_device, smart_inference_mode

class DETRDataset(VajraDetDataset):
    def __init__(self, *args, data=None, task="detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def load_image(self, i, rect_mode=False):
        return super().load_image(i, rect_mode=rect_mode)
    
    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = vajra_transforms(self, self.img_size, hyp, stretch=True)
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
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
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
            img_size = self.args.img_size,
            batch_size = batch,
            augment = False,
            hyp = self.args,
            rect=False,
            cache = self.args.cache or None,
            prefix = colorstr(f"{mode}: "),
            data = self.data
        )
    
    def postprocess(self, preds, model):
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        boxes, logits = preds[0].split((4, nd - 4), dim=-1)
        #logits, boxes = preds["pred_logits"], preds["pred_boxes"]
        #LOGGER.info(f"Num Queries: {model.model.decoder.num_queries}\n")
        #LOGGER.info(f"Num classes: {model.num_classes}")
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= self.args.img_size
        num_queries = self.args.num_queries #model.model.decoder.num_queries
        num_classes = model.num_classes if self.training else model.model.num_classes
        focal = model.loss_config["use_focal_loss"] if self.training else model.model.loss_config["use_focal_loss"]
        if focal == True:
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_queries, dim=-1)
            labels = index % num_classes
            index = index // num_classes
            bbox_pred = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = torch.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > num_queries:
                scores, index = torch.topk(scores, num_queries, dim=-1)
                labels = torch.gather(labels, 1, index)
                bbox_pred = torch.gather(bbox_pred, 1, index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        
        outputs = []
        for lab, box, sco in zip(labels, bbox_pred, scores):
            pred = torch.cat([box, sco.unsqueeze(-1), lab.unsqueeze(-1).float()], dim=-1)
            outputs.append(pred)
        return outputs
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            add_integration_callbacks(self)
            model = Backend(
                weights = model or self.args.model,
                device = select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half
            )

            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            img_size = check_img_size(self.args.img_size, stride=stride)

            if engine:
                self.args.batch = model.batch_size

            elif not pt and not jit:
                self.args.batch = 1
                LOGGER.info(f"Forcing batch=1 square inference (1, 3, {img_size}. {img_size} for non-PyTorch models")
            
            if str(self.args.data).split(".")[-1] in ("yaml", "yml"):
                self.data = check_det_dataset(self.args.data)

            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            
            else:
                raise FileNotFoundError(f"Dataset '{self.args.data}' for task={self.args.task} not found!")

            if self.device.type in ("cpu", "mps"):
                self.args.workers = 0
            
            if not pt:
                self.args.rect = False

            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            model.warmup(img_size=(1 if pt else self.args.batch, 3, img_size, img_size))
        
        self.run_callbacks("on_val_start")
        dt=(
           Profile(device=self.device),
           Profile(device=self.device),
           Profile(device=self.device), 
        )

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            with dt[0]:
                batch = self.preprocess(batch)
            
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            #with dt[2]:
                #if self.training:
                    #self.loss += model.loss(batch, preds)[1]

            with dt[2]:
                LOGGER.info(f"Postprocessing step")
                preds = self.postprocess(preds, model)

            LOGGER.info("Postprocessing done; Time for updating the metrics")
            self.update_metrics(preds, batch)

            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            
            self.run_callbacks("on_val_batch_end")
        
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
                stats["fitness"] = stats["metrics/mAP50-95(Box)"]
            results = {**stats}
            return {k: round(float(v), 5) for k, v in results.items()}
        
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )

            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats
    
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
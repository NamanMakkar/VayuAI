# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import os
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from vajra.dataset import build_dataloader, build_vajra_dataset, build_vajra_small_obj_dataset
from vajra.dataset.converter import coco80_to_coco91_class
from vajra.core.validator import Validator
from vajra import ops
from vajra.models.vajra import detect, pose
from vajra.utils import LOGGER, TQDM, colorstr
from vajra.dataset.utils import check_cls_dataset, check_det_dataset
from vajra.checks import check_img_size
from vajra.checks import check_requirements
from vajra.metrics import ConfusionMatrix, DetectionMetrics, PoseMetrics, box_iou
from vajra.plotting import output_to_target, plot_images
from vajra.ops import Profile
from vajra.nn.backend import Backend
from vajra.callbacks import get_default_callbacks, add_integration_callbacks
from vajra.utils.torch_utils import de_parallel, select_device, smart_inference_mode

class SmallObjDetectionValidator(pose.PoseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "small_obj_detect"
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING! Apple MPS known Pose Detection bug. Recommend 'device=cpu' for Pose Detection models. "
            )

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = deepcopy(de_parallel(trainer.model)).eval()
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

            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            with dt[3]:
                preds = self.postprocess(preds)

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
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
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
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import random
from copy import copy
import numpy as np
import torch.nn as nn

from vajra.dataset import build_dataloader, build_vajra_dataset
from vajra.core.trainer import Trainer
from vajra.models.vajra import detect
from vajra.nn.vajra import DetectionModel
from vajra.utils import LOGGER, RANK
from vajra.plotting import plot_images, plot_labels, plot_results
from vajra.utils.torch_utils import de_parallel, torch_distributed_zero_first

class DetectionTrainer(Trainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build Vajra Dataset"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_vajra_dataset(cfg=self.args, img_path=img_path, batch=batch, data=self.data, mode=mode, rect=mode=="val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in ["train", "val"]
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING! 'rect=True' is incompatible with Dataloader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            size = (
                random.randrange(self.args.img_size * 0.5, self.args.img_size * 1.5 + self.stride)
                // self.stride
                * self.stride
            )
            scale_factor = size / max(imgs.shape[2:])
            if scale_factor != 1:
                new_shape = [
                    math.ceil(x * scale_factor / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=new_shape, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        self.model.num_classes = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = DetectionModel(model_name, num_classes=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    
    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
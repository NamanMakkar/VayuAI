# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import warnings
import random
import time
import torch
from copy import copy, deepcopy
from datetime import datetime, timedelta
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch import distributed as dist
from torch import nn, optim

from vajra.dataset import build_dataloader, build_vajra_dataset, build_vajra_small_obj_dataset
from vajra.core.trainer import Trainer
from vajra.nn.vajra import load_weight
from vajra.models.vajra import detect
from vajra.models.vajra import small_obj_detect
from vajra.distributed import ddp_cleanup, generate_ddp_command
from vajra import callbacks
from vajra.nn.vajra import DetectionModel, PoseModel
from vajra.utils.autobatch import check_train_batch_size
from vajra.configs import get_config, get_save_dir
from vajra.utils import (
    HYPERPARAMS_CFG,
    HYPERPARAMS_CFG_DICT,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    clean_url,
    colorstr,
    yaml_save,
)
from vajra.plotting import plot_images, plot_labels, plot_results
from vajra.utils.torch_utils import de_parallel, torch_distributed_zero_first
from vajra.dataset.utils import check_cls_dataset, check_det_dataset
from vajra.checks import check_model_file_from_stem, check_amp, check_file, check_img_size, print_args
from vajra.utils.torch_utils import (
    TORCH_2_4, EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, one_cycle, init_seeds, strip_optimizer,
    smart_resume, torch_distributed_zero_first, autocast)

class SmallObjDetectionTrainer(Trainer):
    def __init__(self, config=HYPERPARAMS_CFG_DICT, model_configuration=None, _callbacks=None) -> None:
        self.args = get_config(config, model_configuration)
        self.check_resume(model_configuration)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name
        self.weights_dir = self.save_dir / "weights"

        if RANK in {-1, 0}:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))
        
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0

        self.model = check_model_file_from_stem(self.args.model)
        self.args_model = ("-").join(str(Path(self.args.model).stem).split("-")[:-1]) + "-det" + str(Path(self.args.model).suffix)
        self.model_det = check_model_file_from_stem(("-").join(str(Path(self.args.model).stem).split("-")[:-1]) + "-det" + str(Path(self.args.model).suffix))

        self.last, self.best = self.weights_dir / f'last-{("-").join(str(Path(self.args.model).stem).split("-")[:-1]) + "-pose"}.pt', self.weights_dir / f'best-{("-").join(str(Path(self.args.model).stem).split("-")[:-1]) + "-pose"}.pt'
        self.last_det, self.best_det = self.weights_dir / f'last-{str(Path(self.args_model).stem)}.pt', self.weights_dir / f'best-{str(Path(self.args_model).stem)}.pt'

        try:
            if self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in ("yaml", "yml") or self.args.task in (
                "detect",
                "segment",
                "pose",
                "obb",
                "small_obj_detect",
            ):
                self.data = check_det_dataset(self.args.data, add_kpts=True)
                if "yaml_file" in self.data:
                    self.args.data = self.data["yaml_file"]
        except Exception as e:
            raise RuntimeError(f"Dataset '{clean_url(self.args.data)}' error! {e}") from e
        
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None
        self.lf = None
        self.scheduler = None
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def _setup_train(self, world_size):
        self.run_callbacks("on_pretrain_routine_start")
        #LOGGER.info(f"Building PoseModel for training small object detection\n")
        checkpoint = self.setup_model()
        #LOGGER.info(f"Building DetectionModel for validating small object detection\n")
        checkpoint_det = self.setup_det_model()
        self.model = self.model.to(self.device)
        self.model_det = self.model_det.to(self.device)
        self.set_model_attributes()
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".distributed_focal_loss"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.info(
                    f"WARNING! setting 'requires_grad=True' for frozen layer '{k}'. "
                )
                v.requires_grad = True
        
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in (-1, 0):
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup

        if RANK > -1 and world_size > 1:
            dist.broadcast(self.amp, src=0)
        self.amp = bool(self.amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

        grid_size = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.img_size = check_img_size(self.args.img_size, stride=grid_size, floor=grid_size, max_dim=1)
        self.stride=grid_size

        if self.batch_size == -1 and RANK == -1:
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.img_size, self.amp)
        
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )

            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            self.ema_det = ModelEMA(self.model_det)
            if self.args.plots:
                self.plot_training_labels()

        self.accumulate = max(round(self.args.nominal_batch_size / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nominal_batch_size
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nominal_batch_size)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations
        )

        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(checkpoint)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        num_batches = len(self.train_loader)
        num_warmup_iters = max(round(self.args.warmup_epochs * num_batches), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_start_time = time.time()
        self.train_start_time = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image size {self.args.img_size} train, {self.args.img_size} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f'Logging results to {colorstr("bold", self.save_dir)}\n'
            f'Starting training for ' + (f'{self.args.time} hours...' if self.args.time else f"{self.epochs} epochs...")
        )

        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * num_batches
            self.plot_idx.extend([base_idx, base_idx+1, base_idx+2])
        epoch = self.start_epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=num_batches)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                num_iters = i + num_batches * epoch
                if num_iters <= num_warmup_iters:
                    x_interp = [0, num_warmup_iters]
                    self.accumulate = max(1, int(np.interp(num_iters, x_interp, [1, self.args.nominal_batch_size / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(num_iters, x_interp, [self.args.warmup_bias_lr if j==0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(num_iters, x_interp, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                self.scaler.scale(self.loss).backward()

                if num_iters - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = num_iters

                    if self.args.time:
                        self.stop = (time.time() - self.train_start_time) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                
                memory = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{epoch + 1} / {self.epochs}", memory, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and num_iters in self.plot_idx:
                        self.plot_training_samples(batch, num_iters)
                self.run_callbacks("on_train_batch_end")
                
            self.lr = {f"lr/pg{ir}": x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model, include=["yaml", "num_classes", "args", "names", "stride", "class_weights"])
                self.ema_det.update_attr(self.model_det, include=["yaml", "num_classes", "args", "names", "stride", "class_weights"])

                if (self.args.val and (((epoch + 1) % self.args.val_period == 0) or (self.epochs - epoch) <= 10)) \
                    or final_epoch or self.stopper.possible_stop or self.stop:

                    self.model_det.load(self.model.model)
                    self.metrics, self.fitness = self.validate()

                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                if self.args.time:
                    self.stop |= (time.time() - self.train_start_time) > (self.args.time * 3600)
                    
                if self.args.save or final_epoch:
                    self.save_model()
                    self.save_model_det()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_start_time
            self.epoch_start_time = t

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.args.time:
                    mean_epoch_time = (t - self.train_start_time) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch
                    self.stop |= epoch >= self.epochs
                self.scheduler.step()
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break

            epoch += 1
            
        if RANK in (-1, 0):
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_start_time) / 3600:.3f} hours."  
            )

            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_model_det(self):
        import pandas as pd

        metrics = {**self.metrics, **{"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}
        checkpoint = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model_det)).half(),
            "ema": deepcopy(self.ema_det.ema).half(),
            "updates": self.ema_det.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0"
        }

        torch.save(checkpoint, self.last_det)
        if self.best_fitness == self.fitness:
            torch.save(checkpoint, self.best_det)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(checkpoint, self.weights_dir / f"epoch{self.epoch}-{('-').join(str(Path(self.args.model).stem).split('-')[:-1]) + '-det'}.pt")

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build Vajra Small Object Detection Dataset"""
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
        self.model_det.num_classes = self.data["nc"]
        self.model_det.names = self.data["names"]
        self.model_det.args = self.args
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = PoseModel(model_name, num_classes=self.data["nc"], verbose=verbose and RANK == -1, data_kpt_shape=self.data["kpt_shape"])
        if weights:
            model.load(weights)
        return model

    def setup_model(self):    
        if isinstance(self.model, torch.nn.Module):
            return

        model, weights = self.model, None
        checkpoint = None
        if str(model).endswith(".pt"):
            weights, checkpoint = load_weight(model)
        self.model = self.get_model(model_name=self.args.model, weights=weights, verbose=RANK == -1)
        return checkpoint

    def setup_det_model(self):    
        if isinstance(self.model_det, torch.nn.Module):
            return

        model, weights = self.model_det, None
        checkpoint = None
        if str(model).endswith(".pt"):
            weights, checkpoint = load_weight(model)
        self.model_det = self.get_det_model(model_name=self.args_model, weights=weights, verbose=RANK == -1)
        return checkpoint

    def get_det_model(self, model_name=None, weights=None, verbose=True):
        model = DetectionModel(model_name, num_classes=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return small_obj_detect.SmallObjDetectionValidator(
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
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    
    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
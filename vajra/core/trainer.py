# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import os
import sys
import time
import math
import random
import subprocess
import warnings
import numpy as np
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch import distributed as dist
from torch import nn, optim
import yaml
from vajra.distributed import ddp_cleanup, generate_ddp_command
from tqdm import tqdm
from vajra.nn.vajra import load_weight, load_ensemble_weights
from vajra.new_optimizers.lion import Lion
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
from vajra.checks import check_model_file_from_stem, check_amp, check_file, check_img_size, print_args
from vajra import callbacks
from vajra.utils.files import get_latest_run
from vajra.dataset.utils import check_cls_dataset, check_det_dataset
from vajra.utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, one_cycle, init_seeds, strip_optimizer, autocast,
    smart_resume, torch_distributed_zero_first)

class Trainer:
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
        self.last, self.best = self.weights_dir / f'last-{str(Path(self.args.model).stem)}.pt', self.weights_dir / f'best-{str(Path(self.args.model).stem)}.pt'
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0

        self.model = check_model_file_from_stem(self.args.model)

        try:
            if self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in ("yaml", "yml") or self.args.task in (
                "detect",
                "segment",
                "pose",
                "obb",
            ):
                self.data = check_det_dataset(self.args.data)
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

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning("WARNING! 'rect=True' is incompatible with Multi-GPU training, setting 'rect = False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning("WARNING! 'batch=-1' for Autobatch is incompatible with Multi-GPU training, setting"
                               "default 'batch=16'"
                )
                self.args.batch = 16

            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)

        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),
            rank = RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        self.run_callbacks("on_pretrain_routine_start")
        #LOGGER.info(f"Model Name: {str(self.model)}")
        checkpoint = self.setup_model()
        self.model = self.model.to(self.device)
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
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
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
                #LOGGER.info(f'Bboxes: {batch["bboxes"]}\n')
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
                    #LOGGER.info(f"Image device: {batch['img'].device}\n")
                    #LOGGER.info(f"Bboxes device: {batch['bboxes'].device}\n")
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

                if (self.args.val and (((epoch + 1) % self.args.val_period == 0) or (self.epochs - epoch) <= 10)) \
                    or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                if self.args.time:
                    self.stop |= (time.time() - self.train_start_time) > (self.args.time * 3600)
                    
                if self.args.save or final_epoch:
                    self.save_model()
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

    def save_model(self):
        import pandas as pd

        metrics = {**self.metrics, **{"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}
        checkpoint = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0"
        }

        torch.save(checkpoint, self.last)
        if self.best_fitness == self.fitness:
            torch.save(checkpoint, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(checkpoint, self.weights_dir / f"epoch{self.epoch}-{str(Path(self.args.model).stem)}.pt")

    @staticmethod
    def get_dataset(data):
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):    
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        #LOGGER.info(f"Model name: {str(model)}\n")
        checkpoint = None
        if str(model).endswith(".pt"):
            weights, checkpoint = load_weight(model)
        self.model = self.get_model(model_name=self.args.model, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return checkpoint

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        return batch

    def validate(self):
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, model_name=None, weights=None, verbose=True):
        raise NotImplementedError("This task trainer doesn't support loading config files")

    def get_validator(self):
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        pass

    def progress_string(self):
        return ""

    def plot_training_samples(self, batch, num_iters):
        pass

    def plot_training_labels(self):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        num_cols = len(metrics) + 1
        string = "" if self.csv.exists() else (("%23s," * num_cols % tuple(["epoch"] + keys)).rstrip(",") + "\n")
        with open(self.csv, "a") as f:
            f.write(string + ("%23.5g," * num_cols % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        pass

    def on_plot(self, name, data=None):
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

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

    def check_resume(self, model_configuration):
        resume = self.args.resume

        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                checkpoint_args = load_ensemble_weights(last).args
                if not Path(checkpoint_args["data"]).exists():
                    checkpoint_args["data"] = self.args.data

                resume = True
                self.args = get_config(checkpoint_args)
                self.args.model = str(last)

                for k in "img_size", "batch":
                    if k in model_configuration:
                        setattr(self.args, k, model_configuration[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from,"
                    "i.e. 'vajra train resume model=path/to/last.pt'"
                ) from e
        
        self.resume = resume

    def resume_training(self, checkpoint):
        if checkpoint is None or not self.resume:
            return
        
        best_fitness = 0.0
        start_epoch = checkpoint["epoch"] + 1

        if checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            best_fitness = checkpoint["best_fitness"]

        if self.ema and checkpoint.get("ema"):
            self.ema.ema.load_state_dict(checkpoint["ema"].float().state_dict())
            self.ema.updates = checkpoint["updates"]

        if self.resume:
            assert start_epoch > 0, (
                f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
                f"Start a new training without resuming, i.e. 'vajra train model={self.args.model}'"
            )
            LOGGER.info(
                f"Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs"
            )
        
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {checkpoint['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += checkpoint["epoch"]
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        g_bnw, g_w, g_b = [], [], []

        if name == 'auto':
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            num_classes = getattr(model, "num_classes", 80)
            lr_fit = round(0.002 * 5 / (4 + num_classes), 6)
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 1e4 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        for m in model.modules():
            if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
                g_b.append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                g_bnw.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                g_w.append(m.weight)

        if name == 'SGD':
            optimizer = torch.optim.SGD(g_b, lr=lr, momentum=momentum, nesterov=True)
        elif name in ('Adam', 'AdamW', 'Adamax', 'NAdam', 'RAdam'):
            optimizer = getattr(torch.optim, name, torch.optim.AdamW)(g_b, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'LION':
            optimizer = Lion(g_b, lr=lr, betas=(momentum, 0.99), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g_b, lr=lr, momentum=momentum)
        else:
            raise NotImplementedError(f'Optimizer "{name}" not found in list of available optimizers'
                                      f'Valid arguments are - [Adam, AdamW, NAdam, Adamax, RAdam, SGD, LION, RMSProp, auto]'
                                      f'You can request implementation and support for more optimizers on Github issues')

        optimizer.add_param_group({'params': g_w, 'weight_decay': decay}) # Model module weights
        optimizer.add_param_group({'params': g_bnw}) # BatchNorm2d weights
        LOGGER.info(f'{colorstr("optimizer:")} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups '
                    f'{len(g_bnw)} weight(decay=0.0), {len(g_w)} weight(decay={decay}), {len(g_b)} bias(decay=0.0)'
                    )
        del g_bnw, g_w, g_b
        return optimizer
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import os
import math
import torch
from copy import copy
import time
import warnings
import torch.nn as nn
from torch import distributed as dist
import numpy as np
from vajra import callbacks
from pathlib import Path
from vajra.models.vajra.detect import DetectionTrainer
from vajra.utils.autobatch import check_train_batch_size
from vajra.checks import check_model_file_from_stem, check_amp, check_file, check_img_size, print_args
from vajra.new_optimizers import Lion, AdEMAMix, AdEMAMixDistributedShampoo
from vajra.utils import RANK, colorstr, LOGGER, TQDM
from vajra.utils.torch_utils import (
    TORCH_2_4, EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, one_cycle, init_seeds, strip_optimizer, autocast
    )
from vajra.utils import HYPERPARAMS_DETR_CFG_DICT
from ..build import RTDETR_DetModel
from .val import DETRValidator
from vajra.models.utils.lr_scheduler import FlatCosineLRScheduler

class DETRTrainer(DetectionTrainer):
    def __init__(self, config=HYPERPARAMS_DETR_CFG_DICT, model_configuration=None, _callbacks=None):
        super().__init__(config, model_configuration, _callbacks)

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = RTDETR_DetModel(model_name=model_name, num_classes=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def _setup_scheduler(self):
        self.scheduler = FlatCosineLRScheduler(self.optimizer, self.args.lr_gamma, 
                                               iter_per_epoch=len(self.train_loader), total_epochs=self.args.epochs, 
                                               warmup_iter=self.args.warmup_iter, flat_epochs=self.args.flat_epoch, 
                                               no_aug_epochs=self.args.no_aug_epochs)

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

        if ("dfine" or "deim" in self.model.model_name_type):
            always_freeze_names = ["model.0.decoder.up", "model.0.decoder.reg_scale"]
        else:
            always_freeze_names = []
        freeze_layer_names = [f"{x}" for x in freeze_list] + always_freeze_names #[f"model.{x}." for x in freeze_list]
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
            if self.args.plots:
                self.plot_training_labels()

        self.accumulate = max(round(self.args.nominal_batch_size / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nominal_batch_size
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nominal_batch_size)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr_enc_dec,
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
                self.optimizer = self.scheduler.step(num_iters, self.optimizer)
                #if num_iters <= num_warmup_iters:
                    #x_interp = [0, num_warmup_iters]
                    #self.accumulate = max(1, int(np.interp(num_iters, x_interp, [1, self.args.nominal_batch_size / self.batch_size]).round()))
                    #for j, x in enumerate(self.optimizer.param_groups):
                        #x["lr"] = np.interp(num_iters, x_interp, [self.args.warmup_bias_lr if j==0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        #if "momentum" in x:
                            #x["momentum"] = np.interp(num_iters, x_interp, [self.args.warmup_momentum, self.args.momentum])

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

                if (self.args.val and ((epoch + 1) >= self.args.val_start) and (((epoch + 1) % self.args.val_period == 0) or (self.epochs - epoch) <= 10)) \
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
            if self.stop or (epoch == self.args.stop_epoch):
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
    
    def build_dataset(self, img_path, mode="train", batch=None):
        return super().build_dataset(img_path, mode, batch)
    
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []

        for i in range(bs):
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
    
    def build_optimizer(self, model, name='AdamW', lr=0.0001, momentum=0.9, decay=0.00001, iterations=100000):
        g_backbone_w, g_backbone_b, g_backbone_bnw, g_enc_dec_w, g_enc_dec_b, g_enc_dec_bnw = [], [], [], [], [], []

        if name == 'auto':
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr_enc_dec={self.args.lr_enc_dec}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            num_classes = getattr(model, "num_classes", 80)
            lr_fit = round(0.002 * 5 / (4 + num_classes), 6)
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 1e4 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0
        
        for m in model.model.backbone.modules():
            if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
                g_backbone_b.append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                g_backbone_bnw.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                g_backbone_w.append(m.weight)
        
        enc_dec_modules = [model.model.encoder.modules(), model.model.decoder.modules()]
        for  mods in enc_dec_modules:
            for m in mods:
                if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
                    g_enc_dec_b.append(m.bias)
                if isinstance(m, nn.BatchNorm2d):
                    g_enc_dec_bnw.append(m.weight)
                elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                    g_enc_dec_w.append(m.weight)

        if name == 'SGD':
            optimizer = torch.optim.SGD(g_enc_dec_b, lr=lr, momentum=momentum, nesterov=True)
        elif name in ('Adam', 'AdamW', 'Adamax', 'NAdam', 'RAdam'):
            optimizer = getattr(torch.optim, name, torch.optim.AdamW)(g_enc_dec_b, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
        elif name == 'LION':
            optimizer = Lion(g_enc_dec_b, lr=lr, betas=(momentum, 0.99), weight_decay=decay)
        #elif name == "ADOPT":
            #optimizer = ADOPT(g_enc_dec_b, lr=lr, betas=(momentum, 0.99), weight_decay=decay)
        elif name == "AdEMAMix":
            optimizer = AdEMAMix(g_enc_dec_b, lr=lr, betas=(momentum, 0.999, 0.9999), weight_decay=decay)
        elif name == "AdEMAMixShampoo":
            optimizer = AdEMAMixDistributedShampoo(g_enc_dec_b, lr=lr, betas=(momentum, 0.999, 0.9999), weight_decay=decay)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g_enc_dec_b, lr=lr, momentum=momentum)
        else:
            raise NotImplementedError(f'Optimizer "{name}" not found in list of available optimizers'
                                      f'Valid arguments are - [Adam, AdamW, NAdam, Adamax, RAdam, SGD, LION, RMSProp, auto]'
                                      f'You can request implementation and support for more optimizers on Github issues')
        
        optimizer.add_param_group({"params": g_backbone_w, "lr": self.args.lr_backbone, "weight_decay": self.args.backbone_weight_decay})
        optimizer.add_param_group({"params": g_backbone_b, "lr": self.args.lr_backbone})
        optimizer.add_param_group({"params": g_enc_dec_w, "lr": lr, "weight_decay": decay})
        optimizer.add_param_group({"params": g_backbone_bnw, "lr": self.args.lr_backbone})
        optimizer.add_param_group({"params": g_enc_dec_bnw, "lr": lr})

        del g_backbone_w, g_backbone_b, g_backbone_bnw, g_enc_dec_w, g_enc_dec_b, g_enc_dec_bnw
        return optimizer
    
    def get_validator(self):
        self.loss_names = self.model.loss_config["losses"]
        return DETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
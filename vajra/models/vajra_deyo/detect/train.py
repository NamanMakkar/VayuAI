# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
from torch import nn
from copy import copy
from .val import DEYODetectionValidator
from vajra.nn.vajra import VajraDEYODetectionModel
from vajra.utils import RANK, colorstr, LOGGER
from vajra.models.vajra.detect import DetectionTrainer
from vajra.new_optimizers import Lion, ADOPT, AdEMAMix, AdEMAMixDistributedShampoo

class DEYODetectionTrainer(DetectionTrainer):
    def get_validator(self):
        self.loss_names = "cls_loss"
        return DEYODetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def get_model(self, model_name=None, weights=None, verbose=True):
        model = VajraDEYODetectionModel(model_name, num_classes=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        g_bnw, g_w, g_b = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
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

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname or "norm" in fullname:  # bias (no decay)
                    g_b.append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g_w.append(param)
                else:  # weight (with decay)
                    g_bnw.append(param)

        if name == 'SGD':
            optimizer = torch.optim.SGD(g_b, lr=lr, momentum=momentum, nesterov=True)
        elif name in ('Adam', 'AdamW', 'Adamax', 'NAdam', 'RAdam'):
            optimizer = getattr(torch.optim, name, torch.optim.AdamW)(g_b, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
        elif name == 'LION':
            optimizer = Lion(g_b, lr=lr, betas=(momentum, 0.99), weight_decay=decay)
        elif name == "ADOPT":
            optimizer = ADOPT(g_b, lr=lr, betas=(momentum, 0.99), weight_decay=decay)
        elif name == "AdEMAMix":
            optimizer = AdEMAMix(g_b, lr=lr, betas=(momentum, 0.999, 0.9999), weight_decay=decay)
        elif name == "AdEMAMixShampoo":
            optimizer = AdEMAMixDistributedShampoo(g_b, lr=lr, betas=(momentum, 0.999, 0.9999), weight_decay=decay)
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

    def progress_string(self):
        return ("\n" + "%11s" * 5) % (
            "Epoch",
            "GPU_mem",
            self.loss_names,
            "Instances",
            "Size",
        )
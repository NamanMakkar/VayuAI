# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import sys
from pathlib import Path

from typing import Union
import inspect
import numpy as np
import torch
import torch.nn as nn
from vajra.configs import get_config, get_save_dir, data_for_tasks
from vajra import checks
from vajra.nn.vajra import get_task, load_weight
from vajra.utils import HYPERPARAMS_CFG_DICT, ASSETS, LOGGER, RANK, SETTINGS, yaml_load
from vajra.callbacks import get_default_callbacks, default_callbacks
from vajra.nn.vajra import load_weight, get_task

class Model(nn.Module):
    def __init__(self,
                 model_name: Union[str, Path] = 'vajra-v1-nano-det.pt',
                 task: str = None,
                 verbose: bool = False
        ) -> None:
        super().__init__()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.checkpoint = None
        self.checkpoint_path = None
        self.callbacks = get_default_callbacks()
        self.config = None
        self.metrics = None
        self.task = task
        self.model_configuration = {}
        self.model_name = str(model_name).strip()

        if not Path(model_name).suffix:
            self._new(model_name=model_name, task=task, verbose=verbose)

        else:
            self.load_model(weights=model_name, task=task)

    def __call__(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.tensor], stream: bool = False, **kwds) -> list:
        return self.predict(source, stream, **kwds)

    def _apply(self, fn) -> "Model":
        self.is_pytorch_file()
        self = super()._apply(fn)
        self.predictor = None
        self.model_configuration['device'] = self.device
        return self

    @staticmethod
    def is_triton_model(model: str) -> bool:
        from urllib.parse import urlsplit
        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    def load_model(self, weights: str, task=None) -> None:
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights)
        weights = checks.check_model_file_from_stem(weights)

        if Path(weights).suffix == ".pt":
            self.model, self.checkpoint = load_weight(weights)
            self.task = self.model.args["task"]
            self.model_configuration = self.model.args = self._reset_checkpoint_args(self.model.args)
            self.checkpoint_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.checkpoint = weights, None
            self.task = task or get_task(weights)
            self.checkpoint_path = weights
        self.model_configuration["model"] = weights
        self.model_configuration["task"] = self.task
        self.model_name = weights

    def _new(self, model_name, task=None, verbose=False):
        self.task = task or get_task(model_name)
        self.model = self._smart_load("model")(model_name=model_name, verbose=verbose and RANK == -1)
        self.model.task = self.task
        self.model_configuration["model"] = self.model_name
        self.model_configuration["task"] = self.task
        self.model.args = {**HYPERPARAMS_CFG_DICT, **self.model_configuration}

    def is_pytorch_file(self) -> None:
        model_pt_path = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        model_pt_module = isinstance(self.model, nn.Module)
        if not (model_pt_path or model_pt_module):
            raise TypeError(f'model = {self.model} SHOULD BE a *.pt PyTorch model, but is a different format.'
                            f'Exported formats like ONNX, TensorRT only support "val" and "predict" modes'
                            f'i.e "vajra predict vajra-v1-nano.onnx"')

    def reset_weights(self) -> "Model":
        self.is_pytorch_file()
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = 'vajra-v1-nano-det.pt') -> "Model":
        self.is_pytorch_file()
        if isinstance(weights, (str, Path)):
            weights, self.checkpoint = load_weight(weights)
        self.model.load(weights)

    def names(self) -> list:
        from vajra.dataset.utils import check_class_names
        return check_class_names(self.model.names) if hasattr(self.model, "names") else None

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:
        self._check_is_pytorch_model()
        from vajra import __version__
        from datetime import datetime

        updates = {
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License",
        }
        torch.save({**self.checkpoint, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):
        self.is_pytorch_file()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        self.is_pytorch_file()
        self.model.fuse()

    def embed(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
              stream: bool = False, **kwargs,) -> list:

        #if not kwargs.get("embed"):
        #    kwargs["embed"] = [len(self.model.model) - 2]
        return self.predict(source, stream, **kwargs)

    def predict(self, 
                source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
                stream: bool = False,
                predictor=None,
                **kwargs,
        ) -> list:

        if source is None:
            source = ASSETS
            LOGGER.warning(f'WARNING! "source" is missing. Using "source = {source}"')
        
        is_cli = (sys.argv[0].endswith("vajra") or sys.argv[0].endswith("vayuvahana")) and any (
            x in sys.argv for x in ("predict", "track", "mode=predict", "mode=track")
        )
        defaults = {"conf" : 0.25, "batch" : 1, "save":is_cli, "mode" : "predict"}
        args = {**self.model_configuration, **defaults, **kwargs}
        prompts = args.pop("prompts", None)
        
        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(model_configuration=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)

        else:
            self.predictor.args = get_config(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None, 
              stream: bool = False, persist: bool = False, **kwargs) -> list:

        if not hasattr(self.predictor, "trackers"):
            from vajra.trackers import register_tracker
            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1
        kwargs["batch"] = kwargs.get("batch") or 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(self, validator=None, **kwargs):
        custom = {"rect":True}
        args = {**self.model_configuration, **custom, **kwargs, "mode":"val"}
        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        return validator.metrics

    def benchmark(self, **kwargs):
        self.is_pytorch_file()
        from vajra.utils.benchmarks import benchmark

        custom = {"verbose": False}
        args = {**HYPERPARAMS_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),
            img_size=args["img_size"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(self, **kwargs):
        self.is_pytorch_file()
        from .exporter import Exporter

        custom={"img_size": self.model.args["img_size"], "batch": 1, "data": None, "verbose": False}
        args = {**self.model_configuration, **custom, **kwargs, "mode":"export"}
        return Exporter(model_configuration=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, trainer=None, **kwargs):
        self.is_pytorch_file()
        checks.check_pip_update_available()
        model_configuration = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.model_configuration
        custom = {"data": HYPERPARAMS_CFG_DICT["data"] or data_for_tasks[self.task]}
        args = {**self.model_configuration, **custom, **kwargs, "mode": "train"}
        if args.get("resume"):
            args["resume"] = self.checkpoint_path

        self.trainer = (trainer or self._smart_load("trainer"))(model_configuration=args, _callbacks=self.callbacks)
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.checkpoint else None, model_name=self.model_name)
            self.model = self.trainer.model

        self.trainer.train()
        if RANK in (-1, 0):
            checkpoint = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = load_weight(checkpoint)
            self.model_configuration = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def tune(self, use_ray=False, iterations=10, *args, **kwargs):
        self.is_pytorch_file()
        if use_ray:
            from vajra.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from vajra.core.tuner import Tuner
            custom = {}
            args = {**self.model_configuration, **custom, **kwargs}
            return Tuner(args=args, _callbacks=self.callbacks)

    @staticmethod
    def _reset_checkpoint_args(args: dict) -> dict:
        include = {"img_size", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        try:
            result = self.task_map[self.task][key]
            return result

        except Exception  as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]
            raise NotImplementedError(
                f'WARNING! "{name}" model does not support "{mode}" mode for "{self.task}" task yet.'
            ) from e

    @property
    def task_map(self) -> dict:
        raise NotImplementedError('Task Map is Model Specific')

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, "transforms") else None
    
    def add_callback(self, event: str, func) -> None:
        self.callback[event].append(func)

    def clear_callback(self, event: str) -> None:
        self.callback[event] = []

    def reset_callbacks(self) -> None:
        for event in default_callbacks.keys():
            self.callbacks[event] = [default_callbacks[event][0]]
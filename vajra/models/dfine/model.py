# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from pathlib import Path

from typing import Union
import numpy as np
import torch
import torch.nn as nn
from vajra.core.model import Model
from vajra.nn.vajra import get_task, load_weight
from vajra.callbacks import get_default_callbacks, default_callbacks
from vajra.utils import RANK, HYPERPARAMS_CFG_DICT
from vajra.models.dfine.detect.val import DETRValidator
from vajra.models.dfine.detect.train import DETRTrainer
from vajra.models.dfine.detect.predict import DETRPredictor
from .build import RTDETR_DetModel

class DETR(Model):
    def __init__(self, model_name: Union[str, Path] = 'dfine-nano-det.pt',
                 task: str = None,
                 verbose: bool = False):
        super().__init__(model_name, task, verbose)
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
            self.load_model(weights=model_name, task=task, verbose=verbose)

    def load_model(self, weights, task=None, verbose=False):
        self.task = task or get_task(weights)
        self.model = self._smart_load("model")(model_name=weights, verbose = verbose and RANK == -1)
        self.model.task = self.task
        self.model_configuration["model"] = weights
        self.model_configuration["task"] = self.task
        self.model.args = {**HYPERPARAMS_CFG_DICT, **self.model_configuration}

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": DETRPredictor,
                "validator": DETRValidator,
                "trainer": DETRTrainer,
                "model": RTDETR_DetModel,
            }
        }
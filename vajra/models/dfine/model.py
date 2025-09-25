# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from pathlib import Path

from typing import Union
import numpy as np
import torch
import torch.nn as nn
from vajra.core.model import Model
from vajra.models.dfine.detect.val import DETRValidator
from vajra.models.dfine.detect.train import DETRTrainer
from vajra.models.dfine.detect.predict import DETRPredictor
from .build import RTDETR_DetModel

class DETR(Model):
    def __init__(self, model_name: Union[str, Path] = 'dfine-nano-det.pt',
                 task: str = None,
                 verbose: bool = False):
        super().__init__(model_name=model_name, task=task, verbose=verbose)

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
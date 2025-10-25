import contextlib
import csv
import urllib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests import MODEL, SOURCE, SOURCES_LIST, TMP
from vajra import DETR, Vajra
from vajra.configs import MODELS, data_for_tasks, tasks
from vajra.dataset.build import load_inference_source
from vajra.utils import (
    ASSETS,
    HYPERPARAMS_CFG,
    HYPERPARAMS_CFG_PATH,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    is_dir_writeable,
    is_github_action_running,
)
from vajra.utils.downloads import download
from vajra.utils.torch_utils import TORCH_1_9

IS_TMP_WRITEABLE = is_dir_writeable(TMP)

def test_model_forward():
    model = Vajra("vajra-v1-nano-det")
    model(source=None, img_size=32, augment=True)

def test_model_methods():
    model = Vajra(MODEL)

    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map

def test_model_profile():
    from vajra.nn.vajra import DetectionModel
    model = DetectionModel()
    im = torch.randn(1, 3, 64, 64)
    _ = model.predict(im, profile=True)


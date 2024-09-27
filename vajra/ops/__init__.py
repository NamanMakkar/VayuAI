# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import os
import re 
import cv2
import math
import contextlib
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from vajra.utils import LOGGER
from vajra.metrics import batch_probabilistic_iou
from .bbox_segment_ops import *

def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def v10postprocess(preds, max_det, num_classes=80):
    assert(4 + num_classes == preds.shape[-1])
    boxes, scores = preds.split([4, num_classes], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, axis=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index = index.repeat(1, 1, scores.shape[-1]))
    scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
    labels = index % num_classes
    index = index // num_classes
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    return boxes, scores, labels

def find_dataset_yaml_file(path: Path) -> Path:
    files = list(path.glob('*.yaml')) or list(path.rglob('*.yaml'))
    assert files, f'No dataset YAML file found in {path.resolve()}'
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]
    assert len(files) == 1, f'Expected 1 YAML file in {path.resolve()} but found {len(files)}.\n{files}'
    return files[0]

def check_detection_dataset(dataset):
    pass

def convert_torch_to_np_batch(batch: torch.Tensor) -> np.ndarray:
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for oriented bounding boxes, powered by probiou and fast-nms.

    Args:
        boxes (torch.Tensor): (N, 5), xywhr.
        scores (torch.Tensor): (N, ).
        threshold (float): IoU threshold.

    Returns:
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probabilistic_iou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]

class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0, device: torch.device = None):
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt

    def __str__(self):
        return f"Elapsed time is {self.t} s"

    def time(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()
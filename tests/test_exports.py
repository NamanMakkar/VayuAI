import shutil
import uuid
from itertools import product
from pathlib import Path

import pytest

from tests import MODEL, SOURCE
from vajra import Vajra
from vajra.configs import data_for_tasks, models_for_tasks, tasks
from vajra.utils import (
    ARM64, IS_RASPBERRY_PI, LINUX, MACOS, WINDOWS
)
from vajra import checks
from vajra.utils.torch_utils import TORCH_1_9, TORCH_1_13

def test_export_torchscript():
    file = Vajra(MODEL).export(format="torchscript", optimize=False, img_size=32)
    Vajra(file)(SOURCE, img_size=32)

def test_export_onnx():
    file = Vajra(MODEL).export(format="onnx", dynamic=True, img_size=32)
    Vajra(file)(SOURCE, img_size=32)

@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
def test_export_openvino():
    file = Vajra(MODEL).export(format="openvino", img_size=32)
    Vajra(file)(SOURCE, img_size=32)

@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms",
    [
        (task, dynamic, int8, half, batch, nms) for task, dynamic, int8, half, batch, nms in product(
            tasks, [True, False], [True, False], [True, False], [1, 2], [True, False]
        ) if not ((int8 and half) or (task == "classify" and nms))
    ],
)
def test_export_openvino_matrix(task, dynamic, int8, half, batch, nms):
    file = Vajra(models_for_tasks[task]).export(
        format="openvino",
        img_size=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=data_for_tasks[task],
        nms=nms,
    )
    if WINDOWS:
        file = Path(file)
        file = file.rename(file.with_stem(f"{file.stem}-{uuid.uuid4()}"))
    Vajra(file)([SOURCE] * batch, img_size=64 if dynamic else 32)
    shutil.rmtree(file, ignore_errors=True)

@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, simplify, nms",
    [
        (task, dynamic, int8, half, batch, simplify, nms) 
        for task, dynamic, int8, half, batch, simplify, nms in product(
            tasks, [True, False], [False], [False], [1, 2], [True, False], [True, False]
        ) if not ((int8 and half) or (task == "classify" and nms) or (task == "obb" and nms and not TORCH_1_13))
    ],
)
def test_export_onnx_matrix(task, dynamic, int8, half, batch, simplify, nms):
    file = Vajra(models_for_tasks[task]).export(
        format="onnx", img_size=32, dynamic=dynamic, int8=int8, half=half, batch=batch, simplify=simplify, nms=nms
    )
    Vajra(file)([SOURCE] * batch, img_size=64 if dynamic else 32)
    Path(file).unlink()

@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms",
    [
        (task, dynamic, int8, half, batch, nms) for task, dynamic, int8, half, batch, nms in product(
            tasks, [False], [False], [False], [1, 2], [True, False]
        ) if not (task == "classify" and nms)
    ],
)
def test_export_torchscript_matrix(task, dynamic, int8, half, batch, nms):
    file = Vajra(models_for_tasks[task]).export(
        format="torchscript", img_size=32, dynamic=dynamic, int8=int8, half=half, batch=batch, nms=nms
    )

    Vajra(file)([SOURCE] * batch, img_size=64 if dynamic else 32)
    Path(file).unlink()

@pytest.mark.slow
@pytest.mark.skipif(not MACOS, reason="CoreML inference only supported on MACOS")
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.12 not supported with PyTorch<=1.8")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="CoreML not supported in Python 3.12")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(tasks, [False], [True, False], [True, False], [1])
        if not (int8 and half)
    ],
)
def test_export_coreml_matrix(task, dynamic, int8, half, batch):
    file = Vajra(models_for_tasks[task]).export(
        format="coreml",
        img_size=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    Vajra(file)([SOURCE] * batch, img_size=32)
    shutil.rmtree(file)

@pytest.slow
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
@pytest.mark.skipif(
    not LINUX or IS_RASPBERRY_PI,
    reason="Test disabled as TF suffers from install conflicts on Windows, macOS and Raspberry PI",
)
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms",
    [
        (task, dynamic, int8, half, batch, nms)
        for task, dynamic, int8, half, batch, nms in product(
            tasks, [False], [True, False], [True, False], [1], [True, False]
        ) if not ((int8 and half) or (task=="classify" and nms))
    ],
)
def test_export_tflite_matrix(task, dynamic, int8, half, batch, nms):
    file = Vajra(models_for_tasks[task]).export(
        format="tflite", img_size=32, dynamic=dynamic, int8=int8, half=half, batch=batch, nms=nms
    )
    Vajra(file)([SOURCE] * batch, img_size=32)
    Path(file).unlink()

@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")
@pytest.mark.skipif(LINUX and ARM64, reason="CoreML not supported on aarch64 Linux")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="CoreML not supported in Python 3.12")
def test_export_coreml():
    if MACOS:
        file = Vajra(MODEL).export(format="coreml", img_size=32)
        Vajra(file)(SOURCE, img_size=32)
    else:
        Vajra(MODEL).export(format="coreml", nms=True, img_size=32)

@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
@pytest.mark.skipif(not LINUX, reason="Test disabled as TF suffers from install conflicts on Windows and MACOS")
def test_export_tflite():
    model = Vajra(MODEL)
    file = model.export(format="tflite", img_size=32)
    Vajra(file)(SOURCE, img_size=32)

@pytest.mark.skipif(True, reason="Test disabled")
@pytest.mark.skipif(not LINUX, reason="TF suffers from install conflicts on Windows and MACOS")
def test_export_pb():
    model = Vajra(MODEL)
    file = model.export(format="pb", img_size=32)
    Vajra(file)(SOURCE, img_size=32)

@pytest.mark.skipif(True, reason="Test disabled as Paddle protobuf and ONNX protobuf requirements conflict.")
def test_export_paddle():
    Vajra(MODEL).export(format="paddle", img_size=32)

@pytest.mark.slow
@pytest.mark.skipif(IS_RASPBERRY_PI, reason="MNN not supported on Raspberry Pi")
def test_export_mnn():
    file = Vajra(MODEL).export(format="mnn", img_size=32)
    Vajra(file)(SOURCE, img_size=32)

@pytest.mark.slow
def test_export_ncnn():
    file = Vajra(MODEL).export(format="ncnn", img_size=32)
    Vajra(file)(SOURCE, img_size=32)

@pytest.mark.skipif(True, reason="Test disabled as keras and tensorflow version conflicts with tflite export")
@pytest.mark.skipif(not LINUX or MACOS, reason="Skipping test on Windows and MACOS")
def test_export_imx():
    model = Vajra("vajra-v1-nano-det.pt")
    file = model.export(format="imx", img_size=32)
    Vajra(file)(SOURCE, img_size=32)
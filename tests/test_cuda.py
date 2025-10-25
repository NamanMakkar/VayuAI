from itertools import product
from pathlib import Path

import pytest
import torch

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
from vajra import Vajra
from vajra.configs import data_for_tasks, models_for_tasks, tasks
from vajra.utils import ASSETS, WEIGHTS_DIR
from vajra.checks import check_amp

def test_checks():
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT

@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_amp():
    model = Vajra("vajra-v1-nano-det.pt").model.cuda()
    assert check_amp(model)

@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [
        (task, dynamic, int8, half, batch) for task, dynamic, int8, half, batch in product(tasks, [True], [True], [False], 2)
        if not (int8 and half)
    ],
)
def test_export_engine_matrix(task, dynamic, int8, half, batch):
    file = Vajra(models_for_tasks[task]).export(
        format="engine",
        img_size=32, 
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=data_for_tasks[task],
        workspace=1,
        simplify=True
    )
    Vajra(file)([SOURCE] * batch, img_size=64 if dynamic else 32)
    Path(file).unlink()
    Path(file).with_suffix(".cache").unlink() if int8 else None

@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_train():
    device = 0 if CUDA_DEVICE_COUNT == 1 else [0, 1]
    Vajra(MODEL).train(data="coco8.yaml", img_size=64, epochs=1, device=device)

@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_predict_multiple_devices():
    model = Vajra("vajra-v1-nano-det.pt")
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    model = model.to("cuda:0")
    assert str(model.device) == "cuda:0"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    model = model.cuda()
    assert str(model.device) == "cuda:0"
    _ = model(SOURCE)
    assert str(model.device) == "cuda:0"

@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_available():
    from vajra.utils.autobatch import check_train_batch_size
    check_train_batch_size(Vajra(MODEL).model.cuda(), img_size=128, amp=True)

@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_utils_benchmarks():
    from vajra.utils.benchmarks import ProfileModels

    Vajra(MODEL).export(format="engine", img_size=32, dynamic=True, batch=1)
    ProfileModels([MODEL], img_size=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()

@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_predict_sam():
    from vajra import SAM
    from vajra.models.sam import Predictor as SAMPredictor

    model = SAM(WEIGHTS_DIR / "sam2.1_b.pt")
    model.info()
    model(SOURCE, device=0)
    model(SOURCE, bboxes=[439, 437, 524, 709], device=0)
    model(ASSETS / "bus.jpg", points=[900, 370], device=0)
    model(ASSETS / "bus.jpg", points=[900, 370], labels=[1], device=0)
    model(ASSETS / "bus.jpg", points=[[900, 370]], labels=[1], device=0)
    model(ASSETS / "bus.jpg", points=[[400, 370], [900, 370]], labels=[1], device=0)
    model(ASSETS / "bus.jpg", points=[[[900, 370], [1000, 100]]], labels=[[1, 1]], device=0)

    overrides = dict(conf=0.25, task="segment", mode="predict", img_size=1024, model=WEIGHTS_DIR / "mobile_sam.pt")
    predictor = SAMPredictor(overrides = overrides)

    predictor.set_image(ASSETS / "bus.jpg")
    predictor.reset_image()
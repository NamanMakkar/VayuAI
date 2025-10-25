import subprocess

import pytest
from PIL import Image

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE
from vajra.configs import data_for_tasks, models_for_tasks, tasks
from vajra.utils import ASSETS, WEIGHTS_DIR
from vajra import checks
from vajra.utils.torch_utils import TORCH_1_9

TASK_MODEL_DATA = [(task, WEIGHTS_DIR / models_for_tasks[task], data_for_tasks[task]) for task in tasks]
MODELS = [WEIGHTS_DIR / models_for_tasks[task] for task in tasks]

def run(cmd):
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    run("vajra help")
    run("vajra checks")
    run("vajra version")
    run("vajra settings reset")
    run("vajra config")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_train(task, model, data):
    run(f"vajra train {task} model={model} data={data} img_size=32 epochs=1 cache=disk")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val(task, model, data):
    run(f"vajra val {task} model={model} data={data} img_size=32 save_txt save_json")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict(task, model, data):
    run(f"vajra predict model={model} task={task} source={ASSETS} img_size=32 save save_crop save_txt")

@pytest.mark.parametrize("model", MODELS)
def test_export(model):
    run(f"vajra export model={model} format=torchscript img_size=32")

@pytest.mark.slow
@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP is not available")
def test_train_gpu(task, model, data):
    run(f"vajra train {task} model={model} data={data} img_size=32 epochs=1 device=0")
    run(f"vajra train {task} model={model} data={data} img_size=32 epochs=1 device=0,1")
import sys
from tests import MODEL

from unittest import mock

from vajra import Vajra
from vajra.configs import get_config
from vajra.core.exporter import Exporter
from vajra.models.vajra import classify, detect, segment
from vajra.utils import ASSETS, WEIGHTS_DIR, HYPERPARAMS_CFG_PATH

def test_func(*args):
    print("callback test passed")

def test_exporter():
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    model = exporter(model=Vajra("vajra-v1-nano-det").model)
    Vajra(model)(ASSETS)

def test_detect():
    overrides = {"data": "coco8.yaml", "model": "vajra-v1-nano-det", "img_size": 32, "epochs": 1, "save": False}
    hyp = get_config(HYPERPARAMS_CFG_PATH)
    hyp.data = "coco8.yaml"
    hyp.img_size=32

    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    val = detect.DetectionValidator(args=hyp)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    pred = detect.DetectionPredictor(overrides={"img_size": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"

    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"
    
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")

def test_segment():
    overrides = {"data": "coco8-seg.yaml", "model": "vajra-v1-nano-seg", "img_size": 32, "epochs": 1, "save": False}
    hyp = get_config(HYPERPARAMS_CFG_PATH)
    hyp.data = "coco8-seg.yaml"
    hyp.img_size = 32
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    val = segment.SegmentationValidator(args=hyp)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    pred = segment.SegmentationPredictor(overrides={"img_size": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_pred_start"], "callback test failed"

    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return
    
    Exception("Resume test failed!")

def test_classify():
    overrides = {"data": "imagenet10", "model": "vajra-v1-nano-cls", "img_size": 32, "epochs": 1, "save": False}
    hyp = get_config(HYPERPARAMS_CFG_PATH)
    hyp.data = "imagenet10"
    hyp.img_size = 32

    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    val = classify.ClassificationValidator(args=hyp)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    pred = classify.ClassificationPredictor(overrides={"img_size": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "predictor test failed"

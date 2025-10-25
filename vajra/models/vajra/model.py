# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
from pathlib import Path
from typing import Any
from vajra.core.model import Model
from vajra.dataset.build import load_inference_source
from vajra.models.vajra import detect, classify, segment, pose, obb, small_obj_detect, world, vajra_e
from vajra.nn.vajra import ClassificationModel, DetectionModel, SegmentationModel, PoseModel, OBBModel, VajraWorld, VajraEModel, VajraESegmentationModel
from vajra.utils import yaml_load, ROOT

class Vajra(Model):
    def __init__(self, model_name = 'vajra-v1-nano-det.pt', task = None, verbose = False):
        path = Path(model_name)
        super().__init__(model_name, task, verbose)
        if "-world" in path.stem:
            new_instance = VajraWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            super().__init__(model_name=model_name, task=task, verbose=verbose)
    
    @property
    def task_map(self) -> dict:
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": classify.ClassificationTrainer,
                "validator": classify.ClassificationValidator,
                "predictor": classify.ClassificationPredictor,
            },

            "multilabel_classify": {
                "model": ClassificationModel,
                "trainer": classify.MultiLabelClassificationTrainer,
                "validator": classify.MultiLabelClassificationValidator,
                "predictor": classify.MultiLabelClassificationPredictor,
            },

            "detect": {
                "model": DetectionModel,
                "trainer": detect.DetectionTrainer,
                "validator": detect.DetectionValidator,
                "predictor": detect.DetectionPredictor,
            },

            "segment":{
                "model": SegmentationModel,
                "trainer": segment.SegmentationTrainer,
                "validator": segment.SegmentationValidator,
                "predictor": segment.SegmentationPredictor,
            },

            "pose":{
                "model": PoseModel,
                "trainer": pose.PoseTrainer,
                "validator": pose.PoseValidator,
                "predictor": pose.PosePredictor,
            },

            "obb":{
                "model": OBBModel,
                "trainer": obb.OBBTrainer,
                "validator": obb.OBBValidator,
                "predictor": obb.OBBPredictor,
            },

            "small_obj_detect":{
                "model": PoseModel,
                "trainer": small_obj_detect.SmallObjDetectionTrainer,
                "validator": small_obj_detect.SmallObjDetectionValidator,
                "predictor": detect.DetectionPredictor,
            },
        }

class VajraWorldModel(Model):
    def __init__(self, model_name = 'vajra-v1-nano-det-world.pt', task = None, verbose = False) -> None:
        super().__init__(model_name, task, verbose)
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": VajraWorld,
                "validator": detect.DetectionValidator,
                "predictor": detect.DetectionPredictor,
                "trainer": world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        self.model.set_classes(classes)
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        if self.predictor:
            self.predictor.model.names = classes

class VajraE(Model):
    def __init__(self, model_name = 'vajrae-v1-small-seg.pt', task = None, verbose = False) -> None:
        super().__init__(model_name, task, verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "model" : VajraEModel,
                "validator": vajra_e.VajraEDetectionValidator,
                "predictor": detect.DetectionPredictor,
                "trainer": vajra_e.VajraETrainer,
            },
            "segment": {
                "model": VajraESegmentationModel,
                "validator": vajra_e.VajraESegmentationTrainer,
                "predictor": segment.SegmentationPredictor,
                "trainer": vajra_e.VajraESegmentationTrainer
            },
        }
    
    def get_text_pe(self, texts):
        assert isinstance(self.model, VajraEModel)
        return self.model.get_text_pe(texts)
    
    def get_visual_pe(self, img, visual):
        assert isinstance(self.model, VajraEModel)
        return self.model.get_visual_pe(img, visual)
    
    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        assert isinstance(self.model, VajraEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        assert isinstance(self.model, VajraEModel)
        return self.model.get_vocab(names)
    
    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        assert isinstance(self.model, VajraEModel)

        if embeddings is None:
            embeddings = self.get_text_pe(classes)
        self.model.set_classes(classes, embeddings)

        assert " " not in classes
        self.model.names = classes

        if self.predictor:
            self.predictor.model.names = classes

    def val(self, validator=None, load_vp: bool = False, refer_data: str | None = None, **kwargs):
        custom = {"rect": not load_vp}
        args = {**self.model_configuration, **custom, **kwargs, "mode": "val"}
        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics
    
    def predict(self, source = None, stream: bool = False, visual_prompts: dict[str, list]={}, refer_image=None, predictor=vajra_e.VajraEVPDetectionPredictor, **kwargs):
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"] == len(visual_prompts["cls"])), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    model_configuration={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", None),
                        "img_size": kwargs.get("img_size", self.model_configuration["img_size"]),
                    },
                    _callbacks=self.callbacks
                )
            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].num_classes = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, segment.SegmentationPredictor) else "detect"
                self.predictor = None

        elif isinstance(self.predictor, vajra_e.VajraEVPDetectionPredictor):
            self.predictor = None
        return super().predict(source, stream, **kwargs)
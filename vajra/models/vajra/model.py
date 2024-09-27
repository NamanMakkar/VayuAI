# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from pathlib import Path

from vajra.core.model import Model
from vajra.models.vajra import detect, classify, segment, pose, obb, small_obj_detect, world
from vajra.nn.vajra import ClassificationModel, DetectionModel, SegmentationModel, PoseModel, OBBModel, VajraWorld
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
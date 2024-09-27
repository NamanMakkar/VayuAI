# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from vajra.models import Vajra
from vajra.nn.vajra import VajraDEYODetectionModel
from .detect.val import DEYODetectionValidator
from .detect.train import DEYODetectionTrainer
from .detect.predict import DEYODetectionPredictor

class VajraDEYO(Vajra):
    def __init__(self, model_name='vajra-deyo-v1-nano-det.pt', task=None, verbose=False):
        super().__init__(model_name, task, verbose)

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "model": VajraDEYODetectionModel,
                "trainer": DEYODetectionTrainer,
                "validator": DEYODetectionValidator,
                "predictor": DEYODetectionPredictor,
            }
        }
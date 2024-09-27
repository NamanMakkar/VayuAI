# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from pathlib import Path

from vajra.core.model import Model
from vajra.utils.torch_utils import model_info
from .build import build_sam
from .predict import Predictor

class SAM(Model):
    def __init__(self, model_name="sam_b.pt") -> None:
        if model_name and Path(model_name).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        super().__init__(model_name, task="segment")

    def load_model(self, weights: str, task=None) -> None:
        self.model = build_sam(weights)

    def predict(self, source, stream = False, bboxes=None, points=None, labels=None, **kwargs):
        model_configuration = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        kwargs.update(model_configuration)
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed=False, verbose=True):
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        return {"segment": {"predictor": Predictor}}
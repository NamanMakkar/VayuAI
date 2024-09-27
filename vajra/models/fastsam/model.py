# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from pathlib import Path
from vajra.core.model import Model
from .predict import FastSAMPredictor
from .val import FastSAMValidator

class FastSAM(Model):
    def __init__(self, model_name="FastSAM-x.pt") -> None:
        if str(model_name) == "FastSAM.pt":
            model_name="FastSAM-x.pt"
        assert Path(model_name).suffix not in (".yaml", ".yml", ".py"), "FastSAM models only support pre-trained models."
        super().__init__(model_name, task="segment")

    @property
    def task_map(self):
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
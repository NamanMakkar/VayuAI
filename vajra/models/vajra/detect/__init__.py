# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

__all__ = ["DetectionTrainer", "DetectionValidator", "DetectionPredictor"]
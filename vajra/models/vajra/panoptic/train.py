# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from copy import copy

from vajra.models import vajra
from vajra.nn.vajra import PanopticModel
from vajra.utils import HYPERPARAMS_CFG_DICT, RANK
from vajra.plotting import plot_images, plot_results

class PanopticSegmentationTrainer(vajra.segment.SegmentationTrainer):
    def __init__(self, config=HYPERPARAMS_CFG_DICT, model_configuration=None, _callbacks=None) -> None:
        if model_configuration is None:
            model_configuration = {}
        model_configuration["task"] = "panoptic"
        super().__init__(config, model_configuration, _callbacks)

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = PanopticModel(model_name, channels=3, num_classes=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "fcl_loss", "dic_loss"
        return vajra.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
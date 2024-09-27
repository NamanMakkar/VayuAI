# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from copy import copy

from vajra.models import vajra
from vajra.nn.vajra import SegmentationModel
from vajra.utils import HYPERPARAMS_CFG_DICT, RANK
from vajra.plotting import plot_images, plot_results

class SegmentationTrainer(vajra.detect.DetectionTrainer):
    def __init__(self, config=HYPERPARAMS_CFG_DICT, model_configuration=None, _callbacks=None) -> None:
        if model_configuration is None:
            model_configuration = {}
        model_configuration["task"] = "segment"
        super().__init__(config, model_configuration, _callbacks)

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = SegmentationModel(model_name, channels=3, num_classes=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return vajra.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / F"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    
    def plot_metrics(self):
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)
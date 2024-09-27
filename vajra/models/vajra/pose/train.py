# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from copy import copy
from vajra.models import vajra
from vajra.nn.vajra import PoseModel
from vajra.utils import HYPERPARAMS_CFG, LOGGER
from vajra.plotting import plot_images, plot_results

class PoseTrainer(vajra.detect.DetectionTrainer):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        if model_configuration is None:
            model_configuration = {}
        model_configuration["task"] = "pose"
        super().__init__(config, model_configuration, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING! Apple MPS known Pose Detection bug. Recommend 'device=cpu' for Pose Detection models. "
            )

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = PoseModel(model_name, channels=3, num_classes=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return vajra.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)
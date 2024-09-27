# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
from vajra.dataset import ClassificationDataset, build_dataloader, build_multilabel_cls_dataset
from vajra.core.validator import Validator
from vajra.utils import LOGGER
from vajra.metrics import ClassificationMetrics, ConfusionMatrix, MultiLabelClassificationMetrics
from vajra.plotting import plot_images

class ClassificationValidator(Validator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassificationMetrics()

    def get_desc(self):
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        self.names = model.names
        self.num_classes = len(model.names)
        self.confusion_matrix = ConfusionMatrix(num_classes=self.num_classes, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5])
        self.targets.append(batch["cls"])

    def finalize_metrics(self, *args, **kwargs):
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        print_format = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(print_format % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

class MultiLabelClassificationValidator(ClassificationValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "multi_label_classify"
        self.args.plots = False
        self.metrics = MultiLabelClassificationMetrics()

    def get_desc(self):
        return ("%22s" + "%11s" * 3) % ("classes", "precision", "recall", "f1")

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_multilabel_cls_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val")

    def print_results(self):
        """Print multi-label evaluation metrics."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.precision, self.metrics.recall, self.metrics.f1))

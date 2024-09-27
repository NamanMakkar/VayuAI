# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torchvision

from vajra.dataset import ClassificationDataset, build_dataloader, build_multilabel_cls_dataset
from vajra.core.trainer import Trainer
from vajra.models import vajra
from vajra.nn.vajra import ClassificationModel, load_weight
from vajra.utils import HYPERPARAMS_CFG, LOGGER, RANK, colorstr
from vajra.plotting import plot_images, plot_results
from vajra.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first, de_parallel

class ClassificationTrainer(Trainer):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        if model_configuration is None:
            model_configuration = {}
        model_configuration["task"] = "classify"
        if model_configuration.get("img_size") is None:
            model_configuration["img_size"] = 224
        super().__init__(config, model_configuration, _callbacks)

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = ClassificationModel(model_name, num_classes=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)

        for module in model.modules():
            if not self.args.pretrained and hasattr(module, "reset_parameters"):
                module.reset_parameters()
            if isinstance(module, torch.nn.Dropout) and self.args.dropout:
                module.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad=True
        return model
    
    def setup_model(self):
        if isinstance(self.model, torch.nn.Module):
            return
        
        model, ckpt = str(self.model), None
        if model.endswith(".pt"):
            self.model, ckpt = load_weight(model, device="cpu")
            for p in self.model.parameters():
                p.requires_grad = True
        elif model.endswith(".py"):
            self.model = self.get_model(config=model)
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](weights="IMAGENET1K_V1" if self.args.pretrained else None)
        else:
            raise FileNotFoundError(f"ERROR: model={model} not found locally or online. Please check model name.")
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])

        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=mode=="train", prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)
        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)

        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        
        return loader

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return vajra.classify.ClassificationValidator(self.test_loader, self.save_dir, _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
    
    def plot_training_samples(self, batch, num_iters):
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),
            fname=self.save_dir / f"train_batch{num_iters}.jpg",
            on_plot=self.on_plot
        )

class MultiLabelClassificationTrainer(ClassificationTrainer):
    def __init__(self, cfg=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        if model_configuration is None:
            model_configuration = {}
        model_configuration["task"] = "multilabel_classify"
        if model_configuration.get("img_size") is None:
            model_configuration["img_size"] = 224
        super().__init__(cfg, model_configuration, _callbacks)

    def get_validator(self):
        self.loss_names = ["loss"]
        return vajra.classify.MultiLabelClassificationValidator(
            self.test_loader, self.save_dir, _callbacks=self.callbacks
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        return build_multilabel_cls_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val")
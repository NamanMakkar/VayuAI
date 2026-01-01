from copy import copy, deepcopy
from pathlib import Path

import torch
import itertools

from typing import Any

from vajra.nn.vajra import VajraEModel
from vajra.dataset import VajraConcatDataset, build_vajra_dataset, build_vision_language_dataset
from vajra.dataset.augment import LoadVisualPrompt
from vajra.models.vajra.detect import DetectionTrainer, DetectionValidator
from vajra.utils import HYPERPARAMS_CFG_DICT, LOGGER, RANK, DATASETS_DIR
from vajra.utils.torch_utils import unwrap_model
from vajra.dataset.utils import check_det_dataset

class VajraETrainer(DetectionTrainer):
    def __init__(self, config=HYPERPARAMS_CFG_DICT, model_configuration: dict | None = None, _callbacks=None):
        if model_configuration is None:
            model_configuration = {}
        assert not model_configuration.get("compile"), f"Training with 'model={model_configuration['model']}' requires 'compile=False'"
        model_configuration["overlap_mask"] = False
        super().__init__(config, model_configuration, _callbacks)
    
    def get_model(self, model_name=None, weights=None, verbose=True):
        model = VajraEModel(
            model_name,
            num_classes=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1
        )

        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        return super().get_validator()
    
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)

        return build_vajra_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
    
class VajraEPETrainer(DetectionTrainer):
    def get_model(self, model_name=None, weights=None, verbose=True):
        model = VajraEModel(
            model_name,
            num_classes=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1
        )

        del model.model[-1].savpe
        assert weights is not None, "Pretrained weight must be provided for linear probing."
        if weights:
            model.load(weights)
        
        model.eval()
        names = list(self.data["names"].values())
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe) # fuse text embeddings to classify head
        model.model[-1].branch_cls[0][2] = deepcopy(model.model[-1].branch_cls[0][2]).requires_grad_(True)
        model.model[-1].branch_cls[1][2] = deepcopy(model.model[-1].branch_cls[1][2]).requires_grad_(True)
        model.model[-1].branch_cls[2][2] = deepcopy(model.model[-1].branch_cls[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model

class VajraETrainerFromScratch(VajraETrainer):
    def __init__(self, config=HYPERPARAMS_CFG_DICT, model_configuration = None, _callbacks=None):
        self.text_embeddings = None
        super().__init__(config, model_configuration, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_vajra_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_vajra_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_vision_language_dataset(
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["nc"],
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)
        return VajraConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def preprocess_batch(self, batch):
        batch = DetectionTrainer.preprocess_batch(self, batch)

        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )

        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
    
    def get_dataset(self):
        final_data = {}
        data_yaml = self.args.data
        assert data_yaml.get("train", False), "train dataset not found"
        assert data_yaml.get("val", False), "validation dataset not found"
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in {"train", "val"}:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
                for k in {"img_path", "json_file"}:
                    path = Path(g[k])
                    if not path.exists() and not path.is_absolute():
                        g[k] = str((DATASETS_DIR / g[k]).resolve())
            final_data[s] += grounding_data
        data["val"] = data["val"][0]
        final_data["val"] = final_data["val"][0]
        final_data["nc"] = data["val"]["nc"]
        final_data["names"] = data["val"]["names"]
        final_data["path"] = data["val"]["path"]

        self.data = final_data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.training_data[d["train"]] = d
        return final_data
    
    def plot_training_labels(self):
        pass

    def final_eval(self):
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
    
    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        model = "mobileclip:blt"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existing cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

class VajraEPEFreeTrainer(VajraEPETrainer, VajraETrainerFromScratch):
    def get_validator(self):
        self.loss_names = "box", "cls", "dfl"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def preprocess_batch(self, batch):
        return DetectionTrainer.preprocess_batch(self, batch)
    
    def set_text_embeddings(self, datasets, batch: int):
        pass

class VajraEVPTrainer(VajraETrainerFromScratch):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, VajraConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset
    
    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        if isinstance(self.train_loader.dataset, VajraConcatDataset):
            for d in self.train_loader.dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            self.train_loader.dataset.transforms.append(LoadVisualPrompt())
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import itertools
from vajra.dataset import build_vajra_dataset, build_vision_language_dataset, VajraConcatDataset
from vajra.dataset.utils import check_det_dataset
from vajra.models import vajra
from vajra.nn.vajra import VajraWorld
from vajra.utils import HYPERPARAMS_CFG, RANK
from vajra import checks
from vajra.utils.torch_utils import de_parallel

def on_pretrain_routine_end(trainer):
    if RANK in {-1, 0}:
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)

class WorldTrainer(vajra.detect.DetectionTrainer):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None) -> None:
        if model_configuration is None:
            model_configuration = {}
        super().__init__(config, model_configuration, _callbacks)
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        self.clip = clip

    def get_model(self, model_name=None, weights=None, verbose=True):
        model = VajraWorld(
            model_name,
            channels=3,
            num_classes=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_vajra_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        dataset = [
            build_vajra_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_vision_language_dataset(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
            for im_path in img_path
        ]
        return VajraConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

    def get_dataset(self):
        final_data = {}
        data_yaml = self.args.data
        assert data_yaml.get("train", False), "train dataset not found"
        assert data_yaml.get("val", False), "validation dataset not found"
        data = {k: [check_det_dataset(dataset=d) for d in v.get("vajra_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validation on 1 dataset, got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:
                continue
            d["minival"] = str(d["path"] / d["minival"])
        
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            vision_language_data = data_yaml[s].get("vision_language_data")

            if vision_language_data is None:
                continue

            vision_language_data = vision_language_data if isinstance(vision_language_data, list) else [vision_language_data]

            for dat in vision_language_data:
                assert isinstance(dat, dict), f"Vision Language data should be provided in dict format, but got {type(dat)}"
            
            final_data[s] += vision_language_data
        
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        self.data = final_data
        return final_data["train"], final_data["val"][0]

    def plot_training_labels(self):
        pass

    def final_eval(self):
        val = self.args.data["val"]["vajra_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        texts = list(itertools.chain(*batch["texts"]))
        text_token = self.clip.tokenize(texts).to(batch["img"].device)
        txt_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)  # torch.float32
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
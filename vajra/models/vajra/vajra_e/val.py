from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from vajra.dataset import VajraConcatDataset, build_dataloader, build_vajra_dataset
from vajra.dataset.augment import LoadVisualPrompt
from vajra.dataset.utils import check_det_dataset
from vajra.models.vajra.detect import DetectionValidator
from vajra.models.vajra.segment import SegmentationValidator
from vajra.nn.head import PEDetection
from vajra.nn.vajra import VajraEModel
from vajra.utils import LOGGER, TQDM
from vajra.utils.torch_utils import select_device, smart_inference_mode

class VajraEDetectionValidator(DetectionValidator):
    @smart_inference_mode()
    def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: VajraEModel) -> torch.Tensor:
        assert isinstance(model, VajraEModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count
        
        cls_visual_num = cls_visual_num.to(self.device)
        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)

        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: cls.shape[0]] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)
    
    def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader:
        dataset = build_vajra_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, VajraConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )
    
    @smart_inference_mode()
    def __call__(self, 
                 trainer: Any | None = None, 
                 model: VajraEModel | str | None = None,
                 refer_data: str | None = None,
                 load_vp: bool = False
    ) -> dict[str, Any]:
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device, verbose=False)

            if isinstance(model, (str, Path)):
                from vajra.nn.vajra import load_weight

                model, _ = load_weight(model, device=self.device)
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                dataloader = self.get_vpe_dataloader(data)
                vpe = self.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], PEDetection) and hasattr(model.model[-1], "lrpc"):
                return super().__call__(trainer, model)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats
    
class VajraESegmentationValidator(VajraEDetectionValidator, SegmentationValidator):
    pass
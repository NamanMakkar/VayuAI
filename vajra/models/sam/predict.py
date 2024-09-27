# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from vajra.dataset.augment import LetterBox
from vajra.core.predictor import Predictor
from vajra.core.results import Results
from vajra.utils import HYPERPARAMS_CFG
from vajra import ops
from vajra.utils.torch_utils import select_device
from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .build import build_sam

class SamPredictor(Predictor):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        if model_configuration is None:
            model_configuration = {}
        model_configuration.update(dict(task="segment", mode="predict", img_size=1024))
        super().__init__(config, model_configuration, _callbacks)
        self.args.retina_masks = True
        self.img = None
        self.features = None
        self.prompts={}
        self.segment_all = False

    def preprocess(self, img):
        if self.img is not None:
            return self.img
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        if not_tensor:
            img = (img - self.mean) / self.std
        return img

    def pre_transform(self, img):
        assert len(img) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.args.img_size, auto=False, center=False)
        return super().pre_transform(img)

    def inference(self, img, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(img, *args, **kwargs)
        return self.prompt_inference(img, bboxes, points, labels, masks, multimask_output)
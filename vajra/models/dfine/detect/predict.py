# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
from vajra.dataset.augment import LetterBox
from vajra.core.predictor import Predictor
from vajra.core.results import Results
from vajra.ops import *
from vajra.utils import HYPERPARAMS_DETR_CFG

class DETRPredictor(Predictor):
    def __init__(self, config=HYPERPARAMS_DETR_CFG, model_configuration=None, _callbacks=None):
        super().__init__(config, model_configuration, _callbacks)

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):
            orig_imgs = convert_torch_to_np_batch(orig_imgs)

        results = []

        for bbox, score, orig_imgs, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):
            bbox = xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)
            idx = max_score.squeeze(-1) > self.args.conf

            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
            oh, ow = orig_imgs.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_imgs, path=img_path, names=self.model.names, boxes=pred))
        return results
    
    def pre_transform(self, img):
        letterbox = LetterBox(self.img_size, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in img]
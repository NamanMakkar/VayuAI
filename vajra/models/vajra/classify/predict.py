# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import cv2
import torch
from PIL import Image
from vajra.core.predictor import Predictor
from vajra.core.results import Results
from vajra.utils import HYPERPARAMS_CFG
from vajra import ops

class ClassificationPredictor(Predictor):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        super().__init__(config, model_configuration, _callbacks)
        self.args.task = "classify"

    def preprocess(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(
                    im, cv2.COLOR_BGR2RGB
                ))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch_to_np_batch(orig_imgs)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(Results(orig_img, img_path, names=self.model.names, probs=pred)) 
        return results

class MultiLabelClassificationPredictor(Predictor):
    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            results.append(Results(orig_img, path=img_path, names=self.model.names, probs=pred))
        return results
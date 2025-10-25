import numpy as np
import torch

from vajra.dataset.augment import LoadVisualPrompt
from vajra.models.vajra.detect import DetectionPredictor
from vajra.models.vajra.segment import SegmentationPredictor

class VajraEVPDetectionPredictor(DetectionPredictor):
    def setup_model(self, model, verbose=True):
        super().setup_model(model, verbose)
        self.done_warmup = True

    def set_prompts(self, prompts):
        self.prompts = prompts

    def pre_transform(self, img):
        im = super().pre_transform(img)
        bboxes = self.prompts.pop("bboxes", None)
        masks = self.prompts.pop("masks", None)
        category = self.prompts["cls"]
        if len(img) == 1:
            visuals = self._process_single_image(im[0].shape[:2], img[0].shape[:2], category, bboxes, masks)
            prompts = visuals.unsqueeze(0).to(self.device)
        else:
            assert bboxes is not None, f"Expected bboxes, but got {bboxes}!"
            assert isinstance(bboxes, list) and all(isinstance(b, np.ndarray) for b in bboxes), (
                f"Expected list[np.ndarray], but got {bboxes}!"
            )
            assert isinstance(category, list) and all(isinstance(b, np.ndarray) for b in category), {
                f"Expected list[np.ndarray], but got {category}!"
            }
            assert len(img) == len(category) == len(bboxes), (
                f"Expected same length for all inputs, but got {len(img)}vs{len(category)}vs{len(bboxes)}!"
            )
            visuals = [
                self._process_single_image(im[i].shape[:2], img[i].shape[:2], category[i], bboxes[i])
                for i in range(len(im))
            ]
            prompts = torch.nn.utils.rnn.pad_sequence(visuals, batch_first=True).to(self.device)
        self.prompts = prompts.half() if self.model.fp16 else prompts.float()
        return im
    
    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        if bboxes is not None and len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1], src_shape[1])
            bboxes *= gain
            bboxes[..., 0::2] += round((dst_shape[1] - src_shape[1] * gain) / 2 - 0.1)
            bboxes[..., 1::2] += round((dst_shape[0] - src_shape[0] * gain) / 2 - 0.1)

        elif masks is not None:
            resized_masks = super().pre_transform(masks)
            masks = np.stack(resized_masks)
            masks[masks == 114] = 0
        else:
            raise ValueError("Please provide valid bboxes or masks")
        return LoadVisualPrompt().get_visuals(category, dst_shape, bboxes, masks)
    
    def inference(self, img, *args, **kwargs):
        return super().inference(img, vpe=self.prompts, *args, **kwargs)
    
    def get_vpe(self, source):
        self.setup_source(source)
        assert len(self.dataset) == 1, "get_vpe only supports one image!"
        for _, im0s, _ in self.dataset:
            img = self.preprocess(im0s)
            return self.model(img, vpe=self.prompts, return_vpe=True)
        
class VajraEVPSegmentationPredictor(VajraEVPDetectionPredictor, SegmentationPredictor):
    pass
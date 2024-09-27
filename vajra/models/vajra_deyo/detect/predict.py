# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from vajra.models.vajra.detect import DetectionPredictor
import torch
from vajra import ops
from vajra.core.results import Results

class DEYODetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch_to_np_batch(orig_imgs)
        
        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            orig_img = orig_imgs[i]
            bbox = ops.scale_boxes(img.shape[2:], bbox, orig_img.shape)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

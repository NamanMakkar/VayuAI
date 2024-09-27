# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from vajra.core.results import Results
from vajra.models.vajra.detect.predict import DetectionPredictor
from vajra.utils import HYPERPARAMS_CFG, LOGGER
from vajra import ops

class PosePredictor(DetectionPredictor):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        super().__init__(config, model_configuration, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING! Apple MPS known Pose Detection bug. Recommend 'device=cpu' for Pose Detection models. "
            )
    
    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch_to_np_batch(orig_imgs)
        
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from vajra.core.results import Results
from vajra.models.vajra.detect.predict import DetectionPredictor
from vajra.utils import HYPERPARAMS_CFG, LOGGER
from vajra import ops

class SegmentationPredictor(DetectionPredictor):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None):
        super().__init__(config, model_configuration, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch_to_np_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):
                masks=None
            elif self.args.retina_masks:
                #LOGGER.info(f"Img shape = {img.shape[2:]}")
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        
        return results
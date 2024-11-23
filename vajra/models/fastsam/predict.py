# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
from PIL import Image
from vajra.core.results import Results
from vajra.models.fastsam.utils import adjust_bboxes_to_image_border
from vajra.models.vajra.segment.predict import SegmentationPredictor
from vajra.utils import HYPERPARAMS_CFG
from vajra.metrics import box_iou
from vajra import ops, checks

class FastSAMPredictor(SegmentationPredictor):
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None) -> None:
        super().__init__(config, model_configuration, _callbacks)
        #self.args.task = "segment"
        self.prompts={}

    def postprocess(self, preds, img, orig_imgs):
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)

        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(results.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box
        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)
    
    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]

        for result in results:
            masks = results.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = ops.scale_masks(masks[None], result.orig_shape)[0]

            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)

            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])

                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True

            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points

                if labels is None:
                    labels = torch.ones(points.shape[0])

                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(
                    points
                ), f"Expected `labels` got same size as `point`, but got {len(labels)} and {len(points)}"

                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )

                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_imgs, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_imgs.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self._clip_inference(crop_imgs, texts)
                text_idx = torch.argmax(similarity, dim=-1)
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[None] <= int(text_idx)).sum(0)
                idx[text_idx] = True

            prompt_results.append(result[idx])
        
        return prompt_results
    
    def _clip_inference(self, images, texts):
        """
        CLIP Inference

        Args:
            images (List[PIL.Image]): A list of source images and each of these should be PIL.Image type with RGB channel order.
            texts (List[str]): A list of prompt texts and each should be a string object.

        Returns:
            (torch.Tensor): The similarity between given images and texts
        """
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (not hasattr(self, "clip_model")) or (not hasattr(self, "clip_preprocess")):
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        images = torch.stack([self.clip_preprocess(image).to(self.device) for image in images])
        tokenized_text = clip.tokenize(texts).to(self.device)
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features[:, None]).sum(-1)
    
    def set_prompts(self, prompts):
        """Set prompts in advance."""
        self.prompts = prompts
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from vajra.dataset.augment import LetterBox
from vajra.utils import LOGGER, StringOps
from vajra import ops
from vajra.plotting import Annotator, colors, save_one_box
from vajra.utils.torch_utils import smart_inference_mode

class BaseTensor(StringOps):
    def __init__(self, data, orig_shape) -> None:
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)

class Boxes(BaseTensor):
    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (6, 7), f"expected 6 or 7 values but got {n}"
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self):
        return ops.xyxy_to_xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy
    
    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        xywh = ops.xyxy_to_xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh
    
class Masks(BaseTensor):
    def __init__(self, masks, orig_shape) -> None:
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        return [
            ops.scale_segment_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks_to_segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        return [
            ops.scale_segment_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks_to_segments(self.data)
        ]

class Keypoints(BaseTensor):
    @smart_inference_mode()
    def __init__(self, keypoints, orig_shape) -> None:
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:
            mask = keypoints[..., 2] < 0.5
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        return self.data[..., 2] if self.has_visible else None

class Probs(BaseTensor):
    def __init__(self, probs, orig_shape) -> None:
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        return (-self.data).argsort(0)[:5].to_list()

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        return self.data[self.top5]

class OrientedBoundingBox(BaseTensor):
    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (7, 8), f"Expected 7 or 8 values but got {n}"
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        return self.data[:, :5]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        return ops.xywhr2xyxyxyxy(self.xywhr)
    
    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        x1 = self.xyxyxyxy[..., 0].min(1).values
        x2 = self.xyxyxyxy[..., 0].max(1).values
        y1 = self.xyxyxyxy[..., 1].min(1).values
        y2 = self.xyxyxyxy[..., 1].max(1).values
        xyxy = [x1, y1, x2, y2]
        return np.stack(xyxy, axis=-1) if isinstance(self.data, np.ndarray) else torch.stack(xyxy, dim=-1)

class Results(StringOps):
    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OrientedBoundingBox(obb, self.orig_shape) if obb is not None else None

        self.speed = {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        return self._apply("__getitem__", idx)

    def __len__(self):
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, keypoints=None, obb=None):
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OrientedBoundingBox(obb, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        return self._apply("numpy")

    def cuda(self):
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        return self._apply("to", *args, **kwargs)

    def new(self):
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    def plot(
        self, 
        conf=True, 
        line_width=None, 
        font_size=None, 
        font="Arial.ttf", 
        pil=False, 
        img=None, 
        im_gpu=None, 
        kpt_radius=5, 
        kpt_line=True,
        labels=True, 
        boxes=True, 
        masks=True, 
        probs=True, 
        show=False, 
        save=False, 
        filename=None
    ):

        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs

        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),
            example=names
        )

        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .premute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
                idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
                annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        log_string = ""
        probs = self.probs
        boxes = self.boxes

        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("img.jpg")):
        if self.probs is not None:
            LOGGER.warning("WARNING! Classify task does not support `save_crop`. ")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING! OBB task does not support `save_crop`. ")
            return 
        
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        if self.probs is not None:
            LOGGER.warning("WARNING! Classify results do not support the `summary()` method yet. ")
            return

        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            box = {
                "x1": round(row[0] / w, decimals),
                "y1": round(row[1] / h, decimals),
                "x2": round(row[2] / w, decimals),
                "y2": round(row[3] / h, decimals),
            }

            conf = round(row[-2], decimals)
            class_id = int(row[-1])
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": box}
            if self.boxes.is_track:
                result["track_id"] = int(row[-3])  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def tojson(self, normalize=False, decimals=5):
        import json
        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)
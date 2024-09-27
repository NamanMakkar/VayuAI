# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import platform
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from vajra.configs import get_config, get_save_dir
from vajra.dataset import load_inference_source
from vajra.dataset.augment import LetterBox, classify_transforms
from vajra.nn.backend import Backend
from vajra.utils import HYPERPARAMS_CFG, LOGGER, MACOS, WINDOWS, colorstr
from vajra.ops import Profile
from vajra.callbacks import add_integration_callbacks, get_default_callbacks
from vajra.checks import check_img_size, check_imshow
from vajra.utils.files import increment_path
from vajra.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
WARNING! inference results will accumulate in RAM unless `stream-True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and video.

Example:
    results = model(source=..., stream=True)
    for r in results:
        boxes = r.boxes
        masks = r.masks
        probs = r.probs
"""

class Predictor:
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None) -> None:
        self.args = get_config(config, model_configuration)
        self.save_dir = get_save_dir(self.args)

        if self.args.conf is None:
            self.args.conf = 0.25

        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self.model = None
        self.data = self.args.data
        
        self.img_size = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()
        add_integration_callbacks(self)
        
    def preprocess(self, img):
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        if not_tensor:
            img /= 255
        return img

    def inference(self, img, *args, **kwargs):
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(img, augment=self.args.augment, visualize=visualize, *args, **kwargs)

    def pre_transform(self, img):
        same_shapes = len({x.shape for x in img}) == 1
        letterbox = LetterBox(self.img_size, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in img]

    def postprocess(self, preds, img, orig_imgs):
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))

    def predict_cli(self, source=None, model=None):
        gen = self.stream_inference(source, model)
        for _ in gen:
            pass

    def setup_source(self, source):
        self.img_size = check_img_size(self.args.img_size, stride=self.model.stride, min_dim=2)
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.img_size[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )

        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )

        self.source_type = self.dataset.source_type

        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000
            or any(getattr(self.dataset, "video_flag", [False]))
        ):
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        if self.args.verbose:
            LOGGER.info("")

        if not self.model:
            self.setup_model(model)

        with self._lock:
            self.setup_source(source if source is not None else self.args.source)

            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            if not self.done_warmup:
                self.model.warmup(img_size=(1 if self.model.pt or self.model.triton else self.dataset.batch_size, 3, *self.img_size))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None

            profilers = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device)
            )

            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, img0s, s = self.batch

                with profilers[0]:
                    img = self.preprocess(img0s)

                with profilers[1]:
                    preds = self.inference(img, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds
                        continue

                with profilers[2]:
                    self.results = self.postprocess(preds, img, img0s)
                self.run_callbacks("on_predict_postprocess_end")

                num_imgs = len(img0s)

                for i in range(num_imgs):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess":profilers[0].dt * 1e3 / num_imgs,
                        "inference":profilers[1].dt * 1e3 / num_imgs,
                        "postprocess":profilers[2].dt * 1e3 / num_imgs
                    }

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), img, s)

                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results
        
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape"
                f"{(min(self.args.batch, self.seen), 3, *img.shape[2:])}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            num_labels = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{num_labels} label{'s' * (num_labels > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose=True):
        self.model = Backend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn = self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose
        )

        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()

    def write_results(self, i, p, img, s):
        string = ""
        if len(img.shape) == 3:
            img = img[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match.group(1)) if match else None
        
        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "%gx%g " % img.shape[2:]
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        string += result.verbose() + f"{result.speed['inference']:.1f}ms"

        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else img[i],
            )
        
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        img = self.plotted_img

        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if save_path not in self.vid_writer:
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWiter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(img.shape[1], img.shape[0]),  # (width, height)
                )

            self.vid_writer[save_path].write(img)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", img)

        else:
            cv2.imwrite(save_path, img)

    def show(self, p=""):
        img = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(p, img.shape[1], img.shape[0])  # (width, height)
        cv2.imshow(p, img)
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 millisecond

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        self.callbacks[event].append(func)
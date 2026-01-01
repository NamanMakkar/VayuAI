# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import gc
import json
import os
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from vajra.configs import get_config
from vajra.dataset.dataset import VajraDetDataset
from vajra.dataset.build import build_dataloader
from vajra.dataset.utils import check_det_dataset, check_cls_dataset, check_class_names, default_class_names
from vajra.ops import TorchNMS, batch_probabilistic_iou
from vajra.nn.modules import VajraMerudandaBhag1, VajraMerudandaBhag4, VajraV1MerudandaX, VajraV1MerudandaBhag17, SPPFRepViT, VajraV1AttentionBhag8, VajraV1MerudandaBhag16, VajraV1AttentionBhag12, VajraV2AttentionBhag2, VajraV2MerudandaBhag13, VajraV1MerudandaBhag10, VajraV1MerudandaBhag15, VajraV1AttentionBhag11, VajraV2MerudandaBhag14, VajraV2MerudandaBhag15, VajraV1MerudandaBhag8, VajraV1AttentionBhag10, VajraV1AttentionBhag9, VajraV1AttentionBhag1, VajraV1AttentionBhag2, VajraV1AttentionBhag5, VajraV1MakkarNormMerudandaBhag1, VajraV1MakkarNormMerudandaBhag2, VajraV1MerudandaBhag3, VajraV1AttentionBhag4, VajraV1AttentionBhag6, VajraV1MerudandaBhag7, VajraV1AttentionBhag7, VajraV1MerudandaBhag6, VajraV1Attention, VajraV2InnerBlock, VajraMerudandaBhag7, AttentionBottleneckV2, RepNCSPELAN4, VajraV2MerudandaBhag10
from vajra.nn.window_attention import VajraV1SwinTransformerBlockV1, VajraV1SwinTransformerBlockV2, VajraV1SwinTransformerBlockV4
from vajra.nn.head import Detection, Classification, DFINETransformer
from vajra.nn.vajra import DetectionModel, ClassificationModel, SegmentationModel, VajraWorld
from vajra.utils import (
    ARM64,
    HYPERPARAMS_CFG,
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
    ROOT,
    IS_JETSON,
    IS_RASPBERRY_PI,
    WINDOWS,
    TORCH_VERSION,
    __version__,
    colorstr,
    get_default_args,
    yaml_save,
)
from vajra.checks import PYTHON_VERSION, check_img_size, check_is_path_safe, check_requirements, check_version
from vajra.utils.downloads import attempt_download_vajra, get_github_assets, attempt_download_asset
from vajra.utils.files import file_size, spaces_in_path
from vajra.callbacks import get_default_callbacks, add_integration_callbacks
from vajra.ops import Profile
from vajra.configs import data_for_tasks
from vajra.utils.downloads import safe_download
from vajra.utils.torch_utils import TORCH_1_11, TORCH_1_13, TORCH_2_4, TORCH_2_1, TORCH_2_9, get_latest_opset, select_device, smart_inference_mode, onnx_arange_patch

def export_formats():
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize", "half", "nms", "dynamic"]],
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify", "nms"]],
        ["OpenVINO", "openvino", "_openvino_model", True, False, ["batch", "dynamic", "half", "int8", "nms", "fraction"]],
        ["TensorRT", "engine", ".engine", False, True, ["batch", "dynamic", "half", "int8", "simplify", "nms", "fraction"]],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "half", "int8", "nms"]],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True, ["batch", "int8", "keras", "nms"]],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "half", "int8", "nms", "fraction"]],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, []],
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "half", "int8", "nms"]],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"]]
    ]

    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x))) #pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])

def best_onnx_opset(onnx, cuda=False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    version = ".".join(TORCH_VERSION.split(".")[:2])
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    return min(opset, onnx.defs.onnx_opset_version())

def gd_output(gd):
    name_list, input_list = [], []
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)))

def validate_args(format, passed_args, valid_args):
    export_args = ["half", "int8", "dynamic", "keras", "nms", "batch", "fraction"]
    assert valid_args is not None, f"ERROR! valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}
    default_args = get_config(HYPERPARAMS_CFG, custom)
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"ERROR! argument '{arg}' is not supported for format='{format}'"

def try_export(inner_func):
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]

        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.0, "0.0 MB output model size"
            LOGGER.info(f"{prefix} export success! {dt.t:.1f}s, saved as '{f} ({file_size(f):.1f} MB)'")
            return f
        except Exception as e:
            LOGGER.info(f"{prefix} export failure! {dt.t:.1f}s: {e}")
            raise e
    return outer_func

class NMSModel(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
        self.is_tf = self.args.format in frozenset({"saved_model", "tflite", "tfjs"})

    def forward(self, x):
        from functools import partial

        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs=pred.shape[0]
        pred = pred.transpose(-1, -2)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))
        if self.args.dynamic and self.args.batch > 1:
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs) # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                score *= mask
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            multiplier = 8 if self.obb else 1

            if self.args.format == "tflite":
                nmsbox *= multiplier
            else:
                nmsbox = multiplier * nmsbox / torch.Tensor(x.shape[2:], **kwargs).max()
            if not self.args.agnostic_nms:
                end = 2 if self.obb else 4
                cls_offset = cls.reshape(-1, 1).expand(nmsbox.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = partial(
                TorchNMS.fast_nms(
                    use_triu=not(
                        self.is_tf or (self.args.opset or 14) < 14 or (self.args.format == "openvino" and self.args.int8)
                    ),
                    iou_func=batch_probabilistic_iou,
                    exit_early=False,
                )
                if self.obb else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.args.task == "segment" else out[:bs]

class Exporter:
    def __init__(self, config=HYPERPARAMS_CFG, model_configuration=None, _callbacks=None) -> None:
        self.args = get_config(config, model_configuration)
        if self.args.format.lower() in ("coreml", "mlmodel"):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        self.callbacks = _callbacks or get_default_callbacks()
        add_integration_callbacks(self)

    @smart_inference_mode()
    def __call__(self, model=None):
        self.run_callbacks("on_export_start")
        t = time.time()
        format = self.args.format.lower()
        LOGGER.info(f"pt_path for model weights: {getattr(model, 'pt_path', None)}")
        if format in ("tensorrt", "trt"):
            format = "engine"

        if format in ("mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"):
            format = "coreml"
        fmts_dict = export_formats()
        formats = tuple(fmts_dict["Argument"][1:])
        LOGGER.info(f"Formats: {formats}\n")
        if format not in formats:
            import difflib
            matches = difflib.get_close_matches(format, formats, n=1, cutoff=0.6) # 60% similarity required to match
            if not matches:
                message = "Model is already in PyTorch format." if format == "pt" else f"Invalid export format='{format}'."
                raise ValueError(f"{message} Valid formats are {formats}")
            LOGGER.warning(f"Invalid export format='{format}', updating to format='{matches[0]}'")
            format = matches[0]
        flags = [x == format for x in formats]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{format}'. Valid formats are {formats}")
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn, executorch = flags
        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        dla = None
        if format == "engine" and self.args.device is None:
            LOGGER.warning("WARNING! TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if format == "engine" and "dla" in str(self.args.device):
            dla = self.args.device.split(":")[-1]
            self.args.device = "0"
            assert dla in {"0", "1"}, f"Expected self.args.device='dla:0' or 'dla:1' but got {self.args.device}."

        self.device = select_device("cpu" if self.args.device is None else self.args.device)
        format_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(format, self.args, format_keys)
        if not hasattr(model, "names"):
            model.names = default_class_names()
        
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("WARNING! half=True and int8=True are mutually exclusive, setting half=False")
            self.args.half = False
        if self.args.half and onnx and self.device.type == "cpu":
            LOGGER.warning("WARNING! half=True only compatible with GPU export, i.e use device=0")
            self.args.half = False
            assert not self.args.dynamic, "half=True not compatible with dynamic=True, i.e use only one."
        self.img_size = check_img_size(self.args.img_size, stride=model.stride, min_dim=2)
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimizer=True not compatible with cuda devices, i.e. use device='cpu'"
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not tflite or not ARM64 or not LINUX, "TFLite export with NMS unsupported on ARM64 Linux"
            assert not is_tf_format or TORCH_1_13, "Tensorflow exports with NMS require torch>=1.13"
            self.args.conf = self.args.conf or 0.25
        if (engine or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else 'format=engine'}' requires max batch size, i.e. 'batch=16'"
            )
        if edgetpu:
            if not LINUX or ARM64:
                raise SystemError("Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler")
            elif self.args.batch != 1:
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, VajraWorld):
            LOGGER.warning(
                "WARNING! VajraWorld export is not supported to any format yet.\n"
            )

        if self.args.int8 and not self.args.data:
            self.args.data = HYPERPARAMS_CFG.data or data_for_tasks[getattr(model, "task", "detect")]
            LOGGER.warning(
                "WARNING! INT8 export requires a missing 'data' arg for calibration. "
                f"Using default 'data={self.args.data}'."
            )

        img = torch.zeros(self.args.batch, 3, *self.img_size).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or model.model_name
        )

        model = deepcopy(model).to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        for module in model.modules():
            if isinstance(module, (Classification)):
                module.export = True
            if isinstance(module, (Detection, DFINETransformer)):
                module.dynamic = self.args.dynamic
                module.export = True
                module.format = self.args.format
            elif isinstance(module, (VajraV1SwinTransformerBlockV1, VajraV1MerudandaBhag17, VajraV1MerudandaBhag16, VajraV1MerudandaBhag15, SPPFRepViT, VajraV2AttentionBhag2, VajraV1AttentionBhag10, VajraV2MerudandaBhag13, VajraV1MerudandaBhag10, VajraV1AttentionBhag11, VajraV1AttentionBhag12, VajraV1MerudandaBhag8, VajraV1MerudandaBhag7, VajraV1AttentionBhag8, VajraV1AttentionBhag9, VajraV2MerudandaBhag10, VajraV2MerudandaBhag14, VajraV2MerudandaBhag15, VajraV1MakkarNormMerudandaBhag2, VajraV1SwinTransformerBlockV4, VajraMerudandaBhag4, VajraV1MerudandaBhag3, VajraV1MerudandaBhag6, VajraV1AttentionBhag1, VajraV1AttentionBhag2, VajraV1AttentionBhag4, VajraV1AttentionBhag5, VajraV1AttentionBhag6, VajraV1AttentionBhag7, VajraV1Attention, AttentionBottleneckV2, VajraMerudandaBhag7, RepNCSPELAN4, VajraV1MerudandaX)) and not is_tf_format:
                module.forward = module.forward_split

        y = None
        for _ in range(2):
            y = NMSModel(model, self.args)(img) if self.args.nms and not coreml else model(img)

        if self.args.half and onnx and self.device.type != "cpu":
            img, model = img.half(), model.half()

        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        self.img = img
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )

        self.product_path = Path(self.model.model_name)
        name = str(self.product_path.stem).strip("-")[0].replace("vajra", "Vajra")
        version = str(self.product_path.stem).strip("-")[1].replace("v1", "V1")
        self.product_name = name + version + "-" + "-".join(str(self.product_path.stem).split("-")[2:-1])

        #self.product_name = (Path(self.model.model.model_name).stem.split("-")[0].replace("vajra", "Vajra"))
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Vayuvahana Technologies {self.product_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Vayuvahana Technologies Ltd.",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "img_size": self.img_size,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in format_keys}
        }

        if dla is not None:
            self.metadata["dla"] = dla

        if model.task == "pose":
            self.metadata["keypoint_shape"] = model.model[-1].keypoint_shape


        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(img.shape)} BCHW and "
            f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)'
        )

        f = [""] * len(formats)
        if jit or ncnn:
            f[0] = self.export_torchscript()

        if engine:
            f[1] = self.export_engine(dla=dla)
        
        if onnx:
            f[2] = self.export_onnx()
        
        if xml:
            f[3] = self.export_openvino()

        if coreml:
            f[4] = self.export_coreml()

        if any((saved_model, pb, tflite, edgetpu, tfjs)):
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()

            if pb or tfjs:
                f[6] = self.export_pb(keras_model=keras_model)
            
            if tflite:
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)

            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")

            if tfjs:
                f[9] = self.export_tfjs()
        if paddle:
            f[10] = self.export_paddle()
        if ncnn:
            f[11] = self.export_ncnn()
        if executorch:
            f[12] = self.export_executorch()

        f = [str(x) for x in f if x]

        if any(f):
            f = str(Path(f[-1]))
            square = self.img_size[0] == self.img_size[1]
            s = (
                ""
                if square
                else f"WARNING! non-PyTorch val requires square imgages, 'img_size={self.img_size}' will not"
                f"work. Use export 'img_size={max(self.img_size)}' if val is required."
            )

            img_size = self.img_size[0] if square else str(self.img_size)[1:-1].replace(" ", "")
            predict_data = f"data={data}" if model.task == "segment" and format=="pb" else ""
            q = "int8" if self.args.int8 else "half" if self.args.half else ""

            LOGGER.info(
                f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nPredict:         vajra predict task={model.task} model={f} img_size={img_size} {q} {predict_data}'
                f'\nValidate:        vajra val task={model.task} model={f} img_size={img_size} data={data} {q} {s}'
                f'\nVisualize:       https://netron.app'
            )
        
        self.run_callbacks("on_export_end")
        return f

    def get_int8_calibration_dataloader(self, prefix=""):
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)
        dataset = VajraDetDataset(
            data[self.args.split or "val"],
            data=data,
            fraction=self.args.fraction,
            task=self.model.task,
            img_size = self.img_size[0],
            augment=False,
            batch_size=self.args.batch,
        )
        n = len(dataset)
        if n < self.args.batch:
            raise ValueError(
                f"The calibration dataset ({n} images) must have at least as many images as the batch size "
                f"('batch={self.args.batch}')."
            )
        elif n < 300:
            LOGGER.warning(f"{prefix} WARNING! > 300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=self.args.batch, workers=0)
    
    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}")
        f = self.file.with_suffix(".torchscript")
        ts = torch.jit.trace(self.model, self.img, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}

        if self.args.optimize:
            LOGGER.info(f"{prefix} optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(ts).__save_for_lite_interpreter(str(f), _extra_files=extra_files)

        else:
            ts.save(str(f), _extra_files=extra_files)

        return f
    
    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        requirements = ["onnx>=1.12.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime"]
        check_requirements(requirements)
        import onnx

        opset_version = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type) #get_latest_opset()
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"
        
        f = str(self.file.with_suffix(".onnx"))

        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            #self.model.cpu()
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}
            if self.args.nms:
                dynamic["output0"].pop(2)

        if self.args.nms and self.model.task == "obb":
            self.args.opset = opset_version
        
        with onnx_arange_patch(self.args):
            kwargs = {"dynamo": False} if TORCH_2_4 else None
            torch.onnx.export(
                NMSModel(self.model, self.args) if self.args.nms else self.model,
                self.img,
                f,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic or None,
                **kwargs,
            )

        model_onnx = onnx.load(f)

        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)
            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        onnx.save(model_onnx, f)
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={torch.__version__} is installed"
        ov_model = ov.convert_model(
            NMSModel(self.model, self.args) if self.args.nms else self.model,
            input=None if self.args.dynamic else [self.img.shape],
            example_input=self.img,
        )

        def serialize(ov_model, file):
            ov_model.set_rt_info("Vajra-v1", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])

            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])
            
            ov.save_model(ov_model, file, compress_to_fp16=self.args.half)
            yaml_save(Path(file).parent / "metadata.yaml", self.metadata)

        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)
            check_requirements("packaging>=23.2")
            check_requirements("nncf>=2.14.0")
            import nncf

            def transform_fn(data_item):
                data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
                assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
                
                img = data_item.numpy().astype(np.float32) / 255.0
                return np.expand_dims(img, 0) if img.ndim == 3 else img
            
            #LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
            #data = check_det_dataset(self.args.data)
            #dataset = VajraDetDataset(data["val"], data=data, img_size=self.img_size[0], augment=False)
            #num_imgs = len(dataset)

            #if num_imgs < 300:
                #LOGGER.warning(f"{prefix} WARNING! >300 images recommended for INT8 calibration, found {num_imgs} images.")
            #quantization_dataset = nncf.Dataset(dataset, transform_fn)

            ignored_scope=None
            if isinstance(self.model.model[-1], Detection):
                head_module_name = '.'.join(list(self.model.named_modules())[-1][0].split(".")[:2])

                ignored_scope = nncf.IgnoredScope(
                    patterns = [
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.distributed_focal_loss.*",
                    ],
                    types=["Sigmoid"]
                )

            quantized_ov_model = nncf.quantize(
                model=ov_model, 
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn), 
                preset=nncf.QuantizationPreset.MIXED, 
                ignored_scope=ignored_scope
            )

            serialize(quantized_ov_model, fq_ov)
            return fq

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f
    
    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        assert not IS_JETSON, "Jetson Paddle exports not supported yet"
        check_requirements(
            (
                "paddlepaddle-gpu"
                if torch.cuda.is_available()
                else "paddlepaddle==3.0.0"
                if ARM64
                else "paddlepaddle>=3.0.0",
                "x2paddle",
            )
        )
        import x2paddle
        from x2paddle.convert import pytorch2paddle

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.img])
        yaml_save(Path(f) / "metadata.yaml", self.metadata)
        return f

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))
        f_onnx = self.file.with_suffix(".onnx")
        #f_ts = self.file.with_suffix(".torchscript")

        name = Path("pnnx.exe" if WINDOWS else "pnnx")
        pnnx = name if name.is_file() else ROOT / name

        if not pnnx.is_file():
            LOGGER.warning(
                f"{prefix} WARNING! PNNX not found. Attempting to download binary file from "
                "https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in current working directory "
                f"or in {ROOT}. See PNNX repo for full installation instructions."
            )

            system = "macos" if MACOS else "windows" if WINDOWS else "linux-aarch64" if ARM64 else "linux"
            try:
                release, assets = get_github_assets(repo="pnnx/pnnx", retry=True)
                assets = [x for x in assets if f"{system}.zip" in x][0]
                assert isinstance(asset, str), "Unable to retreive PNNX repo assets"
                LOGGER.info(f"{prefix}, successfully found latest PNNX file {asset}")
            except Exception as e:
                release = "20250930"
                asset = f"pnnx-{release}-{system}.zip"
                LOGGER.warning(f"{prefix} PNNX GitHub assets not found: {e}, using default {asset}")
            unzip_dir = safe_download(f"https://github.com/pnnx/pnnx/releases/download/{release}/{asset}", delete=True)
            if check_is_path_safe(Path.cwd(), unzip_dir):
                shutil.move(src=unzip_dir / name, dst=pnnx)
                pnnx.chmod(0o777)
                shutil.rmtree(unzip_dir)
        
        ncnn_args = [
            f"ncnnparam={f / 'model.ncnn.param'}",
            f"ncnnbin={f / 'model.ncnn.bin'}",
            f"ncnnpy={f / 'model_ncnn.py'}"
        ]

        pnnx_args = [
            f'pnnxparam={f / "model.pnnx.param"}',
            f'pnnxbin={f / "model.pnnx.bin"}',
            f'pnnxpy={f / "model_pnnx.py"}',
            f'pnnxonnx={f / "model.pnnx.onnx"}',
        ]

        cmd = [
            str(pnnx),
            str(f_onnx),
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(self.args.half)}",
            f"device={self.device.type}",
            f"inputshape='{[self.args.batch, 3, *self.img_size]}'",
        ]

        f.mkdir(exist_ok=True)
        LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)

        pnnx_files = [x.rsplit("=")[-1] for x in pnnx_args]
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files):
            Path(f_debug).unlink(missing_ok=True)

        yaml_save(f / "metadata.yaml", self.metadata)
        return str(f)

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        mlmodel = self.args.format.lower() == "mlmodel"
        check_requirements("coremltools>=8.0")
        import coremltools as ct

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        if self.args.batch > 1:
            assert self.args.dynamic, (
                "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'."
            )
        if self.args.dynamic:
            assert not self.args.nms, (
                "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
            )
            assert self.model.task != "classify", "'dynamic=True' is not supported for CoreML classification models."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        bias = [0.0, 0.0, 0.0]
        scale = 1 / 255
        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values()))
            model = self.model
        
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.img, mlprogram=not mlmodel) if self.args.nms else self.model
        
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} WARNING! 'nms=True' is only available for Detect models like 'vajra-v1-nano.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model

        ts = torch.jit.trace(model.eval(), self.img, strict=False)  # TorchScript model

        if self.args.dynamic:
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.args.batch, default=1),
                    self.img.shape[1],
                    ct.RangeDim(lower_bound=32, upper_bound=self.img_size[0] * 2, default=self.img_size[0]),
                    ct.RandeDim(lower_bound=32, upper_bound=self.img_size[1] * 2, default=self.img_size[1]),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.img.shape, scale=scale, bias=bias)]

        # Based on apple's documentation it is better to leave out the minimum_deployment target and let that get set
        # Internally based on the model conversion and output type.
        # Setting minimum_depoloyment_target >= iOS16 will require setting compute_precision=ct.precision.FLOAT32.
        # iOS16 adds in better support for FP16, but none of the CoreML NMS specifications handle FP16 as input.
        ct_model = ct.convert(
            ts,
            inputs=inputs,
            classifier_config=classifier_config,
            convert_to="neuralnetwork" if mlmodel else "mlprogram",
        )

        bits, mode = (8, "kmeans") if self.args.int8 else (16, "linear") if self.args.half else (32, None)
        if bits < 32:
            if "kmeans" in mode:
                check_requirements("scikit-learn")  # scikit-learn package required for k-means quantization
            if mlmodel:
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            elif bits == 8:  # mlprogram already quantized to FP16
                import coremltools.optimize.coreml as cto

                op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
                config = cto.OptimizationConfig(global_config=op_config)
                ct_model = cto.palettize_weights(ct_model, config=config)

        if self.args.nms and self.model.task == "detect":
            ct_model = self._pipeline_coreml(ct_model, weights_dir=None if mlmodel else ct_model.weights_dir)

        m = self.metadata  # metadata dict
        ct_model.short_description = m.pop("description")
        ct_model.author = m.pop("author")
        ct_model.license = m.pop("license")
        ct_model.version = m.pop("version")
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} WARNING! CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f


    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        assert self.img.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                cuda_version = torch.version.cuda.split(".")[0]
                check_requirements(f"tensorrt-cu{cuda_version}>7.0.0,<=10.1.0")
            import tensorrt as trt  # noqa

        check_version(trt.__version__, ">=7.0.0", strict=True)  # require tensorrt>=7.0.0
        check_version(trt.__version__, "!=10.1.0")

        self.args.simplify = True

        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(self.args.workspace * (1 << 30)) if self.args.workspace is not None else 0
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0:
            config.max_workspace_size = workspace

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8

        if dla is not None:
            if not IS_JETSON:
                raise ValueError("DLA is only available on NVIDIA Jetson devices")
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
            if not self.args.half and not self.args.int8:
                raise ValueError(
                    "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
                )
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.img.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING! 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)
            max_shape = (*shape[:2], *(int(max(2, self.args.workspace or 2) * d) for d in shape[2:]))
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
                #profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
            config.add_optimization_profile(profile)
            if int8:
                config.set_calibration_profile(profile)

        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            #config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(self,
                             dataset,
                             batch: int,
                             cache: str = "",
                        ) -> None:
                        trt.IInt8Calibrator.__init__(self)
                        self.dataset = dataset
                        self.data_iter = iter(dataset)
                        self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                        self.batch = batch
                        self.cache = Path(cache)

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    return self.algo

                def get_batch_size(self) -> int:
                    return self.batch or 1

                def get_batch(self, names) -> list:
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        return None

                def read_calibration_cache(self) -> bytes:
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()

                def write_calibration_cache(self, cache) -> None:
                    _ = self.cache.write_bytes(cache)

            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch=2 * self.args.batch,
                cache=str(self.file.with_suffix(".cache"))
            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """VajraV1 TensorFlow SavedModel export."""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            #suffix = "-macos" if MACOS else "-aarch64" if ARM64 else "" if cuda else "-cpu"
            version_gt_eq = ">=2.0.0"
            version_lt_eq = "<=2.19.0"
            check_requirements(f"tensorflow{version_gt_eq},{version_lt_eq}")
            import tensorflow as tf  # noqa
        check_requirements(
            (
                "keras",  # required by 'onnx2tf' package
                "tf_keras<=2.19.0",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "ai-edge-litert>=1.2.0" + (",<1.4.0" if MACOS else ""),
                "onnx>=1.12.0",
                "onnx2tf>=1.26.3",
                "onnxslim>=0.1.71",
                "onnxruntime-gpu" if cuda else "onnxruntime",
                "protobuf>=5",
                "tflite_support<=0.4.3" if IS_JETSON else "tflite_support",  # fix ImportError 'GLIBCXX_3.4.29'
                "flatbuffers>=23.5.26,<100",  # update old 'flatbuffers' included inside tensorflow package
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # onnx_graphsurgeon only on NVIDIA
        )

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            message="https://github.com/ultralytics/ultralytics/issues/5161",
        )
        import onnx2tf

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder

        # Pre-download calibration file to fix https://github.com/PINTO0309/onnx2tf/issues/545
        onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
        if not onnx2tf_file.exists():
            attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)

        # Export to ONNX
        self.args.simplify = True
        f_onnx = self.export_onnx()

        # Export to TF
        np_data = None
        if self.args.int8:
            tmp_file = f / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration images file
            verbosity = "info"
            if self.args.data:
                f.mkdir()
                images = [batch["img"].permute(0, 2, 3, 1) for batch in self.get_int8_calibration_dataloader(prefix)]
                images = torch.cat(images, 0).float()
                # mean = images.view(-1, 3).mean(0)  # imagenet mean [123.675, 116.28, 103.53]
                # std = images.view(-1, 3).std(0)  # imagenet std [58.395, 57.12, 57.375]
                np.save(str(tmp_file), images.numpy().astype(np.float32))  # BHWC
                np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]
        else:
            verbosity = "error"

        LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
        onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity=verbosity,
            output_integer_quantized_tflite=self.args.int8,
            quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
            custom_input_op_name_np_data_path=np_data,
            output_signaturedefs=True,
            disable_group_convolution=True,  # for end-to-end model compatibility
            enable_batchmatmul_unfold=True and not self.args.int8,  # for end-to-end model compatibility
        )
        yaml_save(f / "metadata.yaml", self.metadata)  # add metadata.yaml

        # Remove/rename TFLite models
        if self.args.int8:
            tmp_file.unlink(missing_ok=True)
            for file in f.rglob("*_dynamic_range_quant.tflite"):
                file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
            for file in f.rglob("*_integer_quant_with_int16_act.tflite"):
                file.unlink()  # delete extra fp16 activation TFLite files

        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            f.unlink() if "quant_with_int16_act.tflite" in str(f) else self._add_tflite_metadata(file)

        return str(f), tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        f = self.file.with_suffix(".pb")

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        return f

    @try_export
    def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")):
        import tensorflow as tf  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f), None
    
    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):
        LOGGER.info(f"\n{prefix} starting export with ExecuTorch...")
        assert TORCH_2_9, f"ExecuTorch export requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        check_requirements("setuptools<71.0.0")
        check_requirements(("executorch==1.0.0", "flatbuffers"))

        import torch
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        file_directory = Path(str(self.file).replace(self.file.suffix, "_executorch_model"))
        file_directory.mkdir(parents=True, exist_ok=True)

        file_pte = file_directory / self.file.with_suffix(".pte").name
        sample_inputs = (self.img,)

        et_program = to_edge_transform_and_lower(
            torch.export.export(self.model, sample_inputs), partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        with open(file_pte, "wb") as file:
            file.write(et_program.buffer)

        yaml_save(file_directory / "metadata.yaml", self.metadata)
        return str(file_directory)

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        LOGGER.warning(f"{prefix} WARNING! Edge TPU known bug https://github.com/ultralytics/ultralytics/issues/1185")

        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"export only supported on Linux. See {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # sudo installed on system
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
                "sudo apt-get update",
                "sudo apt-get install edgetpu-compiler",
            ):
                subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model

        cmd = f'edgetpu_compiler -s -d -k 10 --out_dir "{Path(f).parent}" "{tflite_model}"'
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
        self._add_tflite_metadata(f)
        return f, None
    
    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """VajraV1 TensorFlow.js export."""
        check_requirements("tensorflowjs")
        if ARM64:
            # Fix error: `np.object` was a deprecated alias for the builtin `object` when exporting to TF.js on ARM64
            check_requirements("numpy==1.23.5")
        import tensorflow as tf
        import tensorflowjs as tfjs  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path

        gd = tf.Graph().as_graph_def()  # TF GraphDef
        with open(f_pb, "rb") as file:
            gd.ParseFromString(file.read())
        outputs = ",".join(gd_output(gd))
        LOGGER.info(f"\n{prefix} output node names: {outputs}")

        quantization = "--quantize_float16" if self.args.half else "--quantize_uint8" if self.args.int8 else ""
        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # exporter can not handle spaces in path
            cmd = (
                "tensorflowjs_converter "
                f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
            )
            LOGGER.info(f"{prefix} running '{cmd}'")
            subprocess.run(cmd, shell=True)

        if " " in f:
            LOGGER.warning(f"{prefix} WARNING! your model may not work correctly with spaces in path '{f}'.")

        # f_json = Path(f) / 'model.json'  # *.json path
        # with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
        #     subst = re.sub(
        #         r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}}}',
        #         r'{"outputs": {"Identity": {"name": "Identity"}, '
        #         r'"Identity_1": {"name": "Identity_1"}, '
        #         r'"Identity_2": {"name": "Identity_2"}, '
        #         r'"Identity_3": {"name": "Identity_3"}}}',
        #         f_json.read_text(),
        #     )
        #     j.write(subst)
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata."""
        import flatbuffers
        try:
            # TFLite Support bug https://github.com/tensorflow/tflite-support/issues/954#issuecomment-2108570845
            from tensorflow_lite_support.metadata import metadata_schema_py_generated as schema  # noqa
            from tensorflow_lite_support.metadata.python import metadata  # noqa
        except ImportError:  # ARM64 systems may not have the 'tensorflow_lite_support' package available
            from tflite_support import metadata  # noqa
            from tflite_support import metadata_schema_py_generated as schema  # noqa

        # Create model info
        model_meta = schema.ModelMetadataT()
        model_meta.name = self.metadata["description"]
        model_meta.version = self.metadata["version"]
        model_meta.author = self.metadata["author"]
        model_meta.license = self.metadata["license"]

        # Label file
        tmp_file = Path(file).parent / "temp_meta.txt"
        with open(tmp_file, "w") as f:
            f.write(str(self.metadata))

        label_file = schema.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = schema.AssociatedFileType.TENSOR_AXIS_LABELS

        # Create input info
        input_meta = schema.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = "Input image to be detected."
        input_meta.content = schema.ContentT()
        input_meta.content.contentProperties = schema.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = schema.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = schema.ContentProperties.ImageProperties

        # Create output info
        output1 = schema.TensorMetadataT()
        output1.name = "output"
        output1.description = "Coordinates of detected objects, class labels, and confidence score"
        output1.associatedFiles = [label_file]
        if self.model.task == "segment":
            output2 = schema.TensorMetadataT()
            output2.name = "output"
            output2.description = "Mask protos"
            output2.associatedFiles = [label_file]

        # Create subgraph info
        subgraph = schema.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output1, output2] if self.model.task == "segment" else [output1]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()

    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):
        import coremltools as ct  # noqa

        LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
        _, _, h, w = list(self.img.shape)  # BCHW

        # Output shapes
        spec = model.get_spec()
        outs = list(iter(spec.description.output))
        if self.args.format == "mlmodel":
            outs[0].type.multiArrayType.shape[:] = self.output_shape[2], self.output_shape[1] - 4
            outs[1].type.multiArrayType.shape[:] = self.output_shape[2], 4
        #out0, out1 = iter(spec.description.output)
        #if MACOS:
            #from PIL import Image

            #img = Image.new("RGB", (w, h))  # w=192, h=320
            #out = model.predict({"image": img})
            #out0_shape = out[out0.name].shape  # (3780, 80)
            #out1_shape = out[out1.name].shape  # (3780, 4)
        #else:  # linux and windows can not run model.predict(), get sizes from PyTorch model output y
            #out0_shape = self.output_shape[2], self.output_shape[1] - 4  # (3780, 80)
            #out1_shape = self.output_shape[2], 4  # (3780, 4)

        # Checks
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        nc = outs[0].type.multiArrayType.shape[-1]
        if len(names) != nc:
            names = {**names, **{i: str(i) for i in range(len(names), nc)}}
        
        assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # check

        # Define output shapes (missing)
        # out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        # out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
        # spec.neuralNetwork.preprocessing[0].featureName = '0'

        # Flexible input shapes
        # from coremltools.models.neural_network import flexible_shape_utils
        # s = [] # shapes
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
        # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
        # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
        # r.add_height_range((192, 640))
        # r.add_width_range((192, 640))
        # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

        # Print
        # print(spec.description)

        # Model from spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)

        # 3. Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = spec.specificationVersion
        for i in range(len(outs)):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        output_names = ["confidence", "coordinates"]
        for i, name in enumerate(output_names):
            nms_spec.description.output[i].name = name

        #output_sizes = [nc, 4]
        for i in range(len(outs)):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = outs[i].type.multiArrayType.shape[-1] #output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = outs[i].type.multArrayType.shape[-1] #output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = outs[0].name  # 1x507x80
        nms.coordinatesInputFeatureName = outs[1].name  # 1x507x4
        nms.confidenceOutputFeatureName = "confidence"
        nms.coordinatesOutputFeatureName = "coordinates"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = self.args.iou
        nms.confidenceThreshold = self.args.conf
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # 4. Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=output_names,
        )
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = spec.specificationVersion
        pipeline.spec.description.metadata.userDefined.update(
            {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
        )

        # Save the model
        model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
        model.input_description["image"] = "Input image"
        model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
        model.input_description["confidenceThreshold"] = (
            f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
        )
        model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
        model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
        LOGGER.info(f"{prefix} pipeline success")
        return model

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


class IOSDetectModel(torch.nn.Module):
    def __init__(self, model, im, mlprogram=True):
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        self.mlprogram=mlprogram
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor(
                [1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h],
                device=next(model.parameters()).device,
            )

    def forward(self, x):
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        if self.mlprogram and self.nc % 80 != 0:
            pad_length = int(((self.nc + 79) // 80) * 80) - self.nc
            cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)
        return cls, xywh * self.normalize
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
from vajra.nn.modules import VajraMerudandaBhag1, VajraMerudandaBhag4, VajraMerudandaBhag7, AttentionBottleneckV2
from vajra.nn.head import Detection
from vajra.nn.vajra import DetectionModel, SegmentationModel, VajraWorld
from vajra.utils import (
    ARM64,
    HYPERPARAMS_CFG,
    LINUX,
    LOGGER,
    MACOS,
    ROOT,
    IS_JETSON,
    IS_RASPBERRY_PI,
    WINDOWS,
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
from vajra.utils.torch_utils import TORCH_1_13, get_latest_opset, select_device, smart_inference_mode

def export_formats():
    import pandas

    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        ["TensorFlow.js", "tfjs", "_web_model", True, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]

    return pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])

def gd_output(gd):
    name_list, input_list = [], []
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)))

def try_export(inner_func):
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]

        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success! {dt.t:.1f}s, saved as '{f} ({file_size(f):.1f} MB)'")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure! {dt.t:.1f}s: {e}")
            raise e
    return outer_func

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

        if format in ("tensorrt", "trt"):
            format = "engine"

        if format in ("mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"):
            format = "coreml"

        formats = tuple(export_formats()["Argument"][1:])
        LOGGER.info(f"Formats: {formats}\n")
        flags = [x == format for x in formats]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{format}'. Valid formats are {formats}")
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn = flags
        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        if format == "engine" and self.args.device is None:
            LOGGER.warning("WARNING! TensroRT requires GPU export")
            self.args.device = "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

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
        if edgetpu and not LINUX:
            raise SystemError("Edge TPU export only supported on Linux.")
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
            if isinstance(module, (Detection)):
                module.dynamic = self.args.dynamic
                module.export = True
                module.format = self.args.format
            elif isinstance(module, (VajraMerudandaBhag4, AttentionBottleneckV2, VajraMerudandaBhag7)) and not is_tf_format:
                module.forward = module.forward_split

        y = None
        for _ in range(2):
            y = model(img)

        if self.args.half and onnx and self.device.type != "cpu":
            img, model = img.half, model.half

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
        }

        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape


        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(img.shape)} BCHW and "
            f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)'
        )

        f = [""] * len(formats)
        if jit or ncnn:
            f[0], _ = self.export_torchscript()

        if engine:
            f[1], _ = self.export_engine()
        
        if onnx:
            f[2], _ = self.export_onnx()
        
        if xml:
            f[3], _ = self.export_openvino()

        if coreml:
            f[4], _ = self.export_coreml()

        if any((saved_model, pb, tflite, edgetpu, tfjs)):
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()

            if pb or tfjs:
                f[6], _ = self.export_pb(keras_model=keras_model)
            
            if tflite:
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)

            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")

            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:
            f[10], _ = self.export_paddle()
        if ncnn:
            f[11], _ = self.export_ncnn()

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
            task=self.model.task,
            imgsz = self.img_size[0],
            augment=False,
            batch_size=self.args.batch * 2,
        )
        n = len(dataset)
        if n < 300:
            LOGGER.warning(f"{prefix} WARNING! > 300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=self.args.batch * 2, workers=0)
    
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

        return f, None
    
    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        requirements = ["onnx>=1.12.0"]
        if self.args.simplify:
            requirements += ["onnxsim==0.1.34", "onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime"]
        check_requirements(requirements)
        import onnx

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")
        f = str(self.file.with_suffix(".onnx"))

        output_names = ["output0", "output1"] if isinstance(self.model, SegmentationModel) else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}
        
        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,
            self.img.cpu() if dynamic else self.img,
            f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic or None,
        )

        model_onnx = onnx.load(f)
        #LOGGER.info(f"ONNX Model: {model_onnx}\n")

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

        onnx.save(model_onnx, f)
        return f, model_onnx

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        check_requirements("openvino>=2024.0.0")
        import openvino as ov

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
        assert TORCH_1_13, f"OpenVINO export requires torch>=1.13.0 but torch=={torch.__version__} is installed"
        ov_model = ov.convert_model(
            self.model.cpu(),
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
            
            ov.runtime.save_model(ov_model, file, compress_to_fp16=self.args.half)
            yaml_save(Path(file).parent / "metadata.yaml", self.metadata)

        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)
            if not self.args.data:
                self.args.data = HYPERPARAMS_CFG.data or "coco128.yaml"
                LOGGER.warning(
                    f"{prefix} WARNING! INT8 export requires a missing 'data' arg for calibration. "
                    f"Using default 'data={self.args.data}'."
                )

            check_requirements("nncf>=2.8.0")
            import nncf

            def transform_fn(data_item):
                assert (
                    data_item["img"].dtype == torch.uint8
                )
                img = data_item["img"].numpy().astype(np.float32) / 255.0
                return np.expand_dims(img, 0) if img.ndim == 3 else img
            
            LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
            data = check_det_dataset(self.args.data)
            dataset = VajraDetDataset(data["val"], data=data, imgsz=self.img_size[0], augment=False)
            num_imgs = len(dataset)

            if num_imgs < 300:
                LOGGER.warning(f"{prefix} WARNING! >300 images recommended for INT8 calibration, found {num_imgs} images.")
            #quantization_dataset = nncf.Dataset(dataset, transform_fn)

            ignored_scope=None

            if isinstance(self.model.model.head, Detection):
                head_module_name = '.'.join(list(self.model.named_modules())[-1][0].split(".")[:2])

                ignored_scope = nncf.IgnoredScope(
                    patterns = [
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.distributed_focal_loss.*",
                    ],
                    type=["Sigmoid"]
                )

            quantized_ov_model = nncf.quantize(
                model=ov_model, 
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn), 
                preset=nncf.QuantizationPreset.MIXED, 
                ignored_scope=ignored_scope
            )

            serialize(quantized_ov_model, fq_ov)
            return fq, None

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f, None
    
    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        check_requirements(("paddlepaddle", "x2paddle"))
        import x2paddle
        from x2paddle.convert import pytorch2paddle

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.img])
        yaml_save(Path(f) / "metadata.yaml", self.metadata)
        return f, None

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        check_requirements("ncnn")
        import ncnn

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))
        f_ts = self.file.with_suffix(".torchscript")

        name = Path("pnnx.exe" if WINDOWS else "pnnx")
        pnnx = name if name.is_file() else ROOT / name

        if not pnnx.is_file():
            LOGGER.warning(
                f"{prefix} WARNING! PNNX not found. Attempting to download binary file from "
                "https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in current working directory "
                f"or in {ROOT}. See PNNX repo for full installation instructions."
            )

            system = "macos" if MACOS else "windows" if WINDOWS else "linux-aarch64" if ARM64 else "linux"
            _, assets = get_github_assets(repo="pnnx/pnnx", retry=True)

            if assets:
                url = [x for x in assets if f"{system}.zip" in x][0]

            else:
                url = f"https://github.com/pnnx/pnnx/releases/download/20240226/pnnx-20240226-{system}.zip"
                LOGGER.warning(f"{prefix} WARNING! PNNX Github assets not found, using default {url}")
            asset = attempt_download_asset(url, repo="pnnx/pnnx", release="latest")
            if check_is_path_safe(Path.cwd(), asset):  # avoid path traversal security vulnerability
                unzip_dir = Path(asset).with_suffix("")
                (unzip_dir / name).rename(pnnx)  # move binary to ROOT
                shutil.rmtree(unzip_dir)  # delete unzip dir
                Path(asset).unlink()  # delete zip
                pnnx.chmod(0o777)
        
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
            str(f_ts),
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(self.args.half)}",
            f"device={self.device.type}",
            f"inputshape='{[self.args.batch, 3, *self.img_size]}'",
        ]

        f.mkdir(exist_ok=True)
        LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)

        pnnx_files = [x.split("=")[-1] for x in pnnx_args]
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files):
            Path(f_debug).unlink(missing_ok=True)

        yaml_save(f / "metadata.yaml", self.metadata)
        return str(f), None

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        mlmodel = self.args.format.lower() == "mlmodel"
        check_requirements('coremltools>=6.0,<=6.2' if mlmodel else 'coremltools>=7.0')
        import coremltools as ct

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        bias = [0.0, 0.0, 0.0]
        scale = 1 / 255
        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values())) if self.args.nms else None
            model = self.model
        
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.img)
        
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} WARNING! 'nms=True' is only available for Detect models like 'vajra-v1-nano.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model

        ts = torch.jit.trace(model.eval(), self.img, strict=False)  # TorchScript model
        ct_model = ct.convert(
            ts,
            inputs=[ct.ImageType("image", shape=self.img.shape, scale=scale, bias=bias)],
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
            if mlmodel:
                # coremltools<=6.2 NMS export requires Python<3.11
                check_version(PYTHON_VERSION, "<3.11", name="Python ", strict=True)
                weights_dir = None
            else:
                ct_model.save(str(f))  # save otherwise weights_dir does not exist
                weights_dir = str(f / "Data/com.apple.CoreML/weights")
            ct_model = self._pipeline_coreml(ct_model, weights_dir=weights_dir)

        m = self.metadata  # metadata dict
        ct_model.short_description = m.pop("description")
        ct_model.author = m.pop("author")
        ct_model.license = m.pop("license")
        ct_model.version = m.pop("version")
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} WARNING! CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f, ct_model


    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        assert self.img.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,<=10.1.0")
                #check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
            import tensorrt as trt  # noqa

        check_version(trt.__version__, ">=7.0.0", strict=True)  # require tensorrt>=7.0.0
        check_version(trt.__version__, "<=10.1.0")

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
        workspace = int(self.args.workspace * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:
            config.max_workspace_size = workspace

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8
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
            max_shape = (*shape[:2], *(max(1, self.args.workspace) * d for d in shape[2:]))
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
                #profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
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

        return f, None

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """YOLOv8 TensorFlow SavedModel export."""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            suffix = "-macos" if MACOS else "-aarch64" if ARM64 else "" if cuda else "-cpu"
            version = ">=2.0.0"
            check_requirements(f"tensorflow{suffix}{version}")
            import tensorflow as tf  # noqa
        check_requirements(
            (
                "keras",  # required by 'onnx2tf' package
                "tf_keras",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "onnx>=1.12.0",
                "onnx2tf",
                "onnxslim>=0.1.31",
                "onnxsim",
                "tflite_support<=0.4.3" if IS_JETSON else "tflite_support",  # fix ImportError 'GLIBCXX_3.4.29'
                "flatbuffers>=23.5.26,<100",  # update old 'flatbuffers' included inside tensorflow package
                "onnxruntime-gpu" if cuda else "onnxruntime",
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
        f_onnx, _ = self.export_onnx()

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
            disable_group_convolution=True,  # for end-to-end model compatibility
            enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
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
        return f, None

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
        """YOLOv8 TensorFlow.js export."""
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
        return f, None

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
        out0, out1 = iter(spec.description.output)
        if MACOS:
            from PIL import Image

            img = Image.new("RGB", (w, h))  # w=192, h=320
            out = model.predict({"image": img})
            out0_shape = out[out0.name].shape  # (3780, 80)
            out1_shape = out[out1.name].shape  # (3780, 4)
        else:  # linux and windows can not run model.predict(), get sizes from PyTorch model output y
            out0_shape = self.output_shape[2], self.output_shape[1] - 4  # (3780, 80)
            out1_shape = self.output_shape[2], 4  # (3780, 4)

        # Checks
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        _, nc = out0_shape  # number of anchors, number of classes
        # _, nc = out0.type.multiArrayType.shape
        assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # check

        # Define output shapes (missing)
        out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
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
        nms_spec.specificationVersion = 5
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = "confidence"
        nms_spec.description.output[1].name = "coordinates"

        output_sizes = [nc, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = out0.name  # 1x507x80
        nms.coordinatesInputFeatureName = out1.name  # 1x507x4
        nms.confidenceOutputFeatureName = "confidence"
        nms.coordinatesOutputFeatureName = "coordinates"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = 0.45
        nms.confidenceThreshold = 0.25
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
            output_features=["confidence", "coordinates"],
        )
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = 5
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
    def __init__(self, model, im):
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)

    def forward(self, x):
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        return cls, xywh * self.normalize
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import glob
import platform
import time
from pathlib import Path

import numpy as np
import torch.cuda

from vajra import Vajra
from vajra.core.exporter import export_formats
from vajra.configs import data_for_tasks, metrics_for_tasks
from vajra.utils import ASSETS, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
from vajra.checks import IS_PYTHON_3_12, check_requirements, check_vajra
from vajra.utils.files import file_size
from vajra.utils.torch_utils import select_device
from vajra.nn.vajra import VajraWorld

def benchmark(
    model=WEIGHTS_DIR / "vajra-v1-nano-det.pt", data=None, img_size=160, half=False, int8=False, device="cpu", verbose=False
):
    import pandas as pd

    pd.options.display.max_columns = 10
    pd.options.display.width = 120

    device = select_device(device, verbose=False)

    if isinstance(model, (str, Path)):
        model = Vajra(model)

    y = []

    t0 = time.time()

    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():
        filename = None
        try:
            if i == 9:
                assert LINUX, "Edge TPU export only supported on Linux"
            elif i == 7:
                assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"
            elif i in {5, 10}:
                assert MACOS or LINUX, "export only supported on macOS and Linux"
            if i in {3, 5}:  # CoreML and OpenVINO
                assert not IS_PYTHON_3_12, "CoreML and OpenVINO not supported on Python 3.12"
            if i in {6, 7, 8, 9, 10}:  # All TF formats
                assert not isinstance(model, VajraWorld), "VajraWorld TensorFlow exports not supported by onnx2tf yet"
            if i in {11}:  # Paddle
                assert not isinstance(model, VajraWorld), "VajraWorld Paddle exports not supported yet"
            if i in {12}:  # NCNN
                assert not isinstance(model, VajraWorld), "VajraWorld NCNN exports not supported yet"
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            if format == "-":
                filename = model.checkpoint_path or model.model_name
                exported_model = model  # PyTorch format
            else:
                filename = model.export(img_size=img_size, format=format, half=half, int8=int8, device=device, verbose=False)
                exported_model = Vajra(filename, task=model.task)
                assert suffix in str(filename), "export failed"

            assert model.task != "pose" or i != 7, "GraphDef Pose inference is not supported"
            assert i not in (9, 10), "inference not supported"  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            exported_model.predict(ASSETS / "bus.jpg", img_size=img_size, device=device, half=half)

            # Validate
            data = data or data_for_tasks[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
            key = metrics_for_tasks[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect
            results = exported_model.val(
                data=data, batch=1, img_size=img_size, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            y.append([name, round(file_size(filename), 1), round(metric, 4), round(speed, 2)])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.warning(f"ERROR! Benchmark failure for {name}: {e}")
            y.append([name, round(file_size(filename), 1), None, None])  # mAP, t_inference
    
    check_vajra(device=device)
    df = pd.DataFrame(y, columns=["Format", "Size (MB)", key, "Inference time (ms/im)"])
    name = Path(model.checkpoint_path).name
    s = f"\nBenchmarks complete for {name} on {data} at img_size={img_size} ({time.time() - t0:.2f}s)\n{df}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics=df[key].array
        floor=verbose
        assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df

class ProfileModels:
    def __init__(self, 
                 paths: list, 
                 num_timed_runs=100, 
                 num_warmup_runs=10, 
                 min_time=60, 
                 img_size=640, 
                 half=True, 
                 trt=True, 
                 device=None):
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.img_size = img_size
        self.half = half
        self.trt = trt
        self.device = device or torch.device(0 if torch.cuda.is_available() else "cpu")
    
    def profile(self):
        files = self.get_files()

        if not files:
            print("No matching *.pt or *.onnx files found.")
            return
        
        table_rows = []
        output = []

        for file in files:
            engine_file = file.with_suffix(".engine")
            if not file.suffix:
                model = Vajra(str(file))
                model.fuse()
                model_info = model.info()

                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    engine_file = model.export(
                        format="engine", half=self.half, img_size=self.img_size, device=self.device, verbose=False
                    )
                onnx_file = model.export(
                    format="onnx", half=self.half, img_size=self.img_size, simplify=True, device=self.device, verbose=False
                )
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue
            
            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)
        return output
    
    def get_files(self):
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif not path.suffix:
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        print(f"Profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    def get_onnx_model_info(self, onnx_file: str):
        return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def iterative_sigma_clipping(data, sigma=2, max_iters=3):
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        model = Vajra(engine_file)
        input_data = np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)

        elapsed = 0.0

        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, img_size=self.img_size, verbose=False)
            elapsed = time.time() - start_time
        
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)
        run_times = []

        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, img_size=self.img_size, verbose=False)
            run_times.append(results[0].speed["inference"])

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)
        return np.mean(run_times), np.std(run_times)

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        check_requirements("onnxruntime")
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        input_tensor = sess.get_inputs()[0]
        input_type = input_tensor.type

        if "float16" in input_type:
            input_dtype = np.float16
        elif "float" in input_type:
            input_dtype = np.float32
        elif "double" in input_type:
            input_dtype = np.int64
        elif "int32" in input_type:
            input_dtype = np.int32
        else:
            raise ValueError(f"Unsupported ONNX datatype {input_type}")
        
        input_data = np.random.rand(*input_tensor.shape).astype(input_dtype)
        input_name = input_tensor.name
        output_name = sess.get_outputs()[0].name

        # Warmup runs
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], {input_name: input_data})
            elapsed = time.time() - start_time

        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        # Timed runs
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], {input_name: input_data})
            run_times.append((time.time() - start_time) * 1000)  # Convert to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        layers, params, gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.img_size} | - | {t_onnx[0]:.2f} ± {t_onnx[1]:.2f} ms | {t_engine[0]:.2f} ±"
            f"{t_engine[1]:.2f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )
    
    @staticmethod
    def generate_results_dict(model_name, t_onnx, t_engine, model_info):
        layers, params, gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        header = (
            f"| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | "
            f"Speed<br><sup>{gpu} TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |"
        )
        separator = (
            "|-------------|---------------------|--------------------|------------------------------|"
            "-----------------------------------|------------------|-----------------|"
        )

        print(f"\n\n{header}")
        print(separator)
        for row in table_rows:
            print(row)
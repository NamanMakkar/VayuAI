# TensorRT export for VajraV1 models

The TensorRT export format maximises speed and efficiency for VajraV1 model deployment on NVIDIA GPUs. The TensorRT export format can be used for VajraV1 models for swift and efficient inference on NVIDIA hardware. This guide will give you easy-to-follow steps for the conversion process and help you make the most of NVIDIA GPUs for your application.

## TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt), developed by NVIDIA, is an advanced software development kit (SDK) designed for high-speed deep learning inference. It is well suited for real-time applications like object detection.

This toolkit optimizes deep learning models for NVIDIA GPUs and result in faster and more efficient operations. TensorRT models undergo TensorRT optimization, which includes layer fusion, precision calibration (INT8 and FP16), dynamic tensor memory management, and kernel auto-tuning. Converting deep learning models into the TensorRT format allows developers to realize the potential of NVIDIA GPUs fully.

TensorRT is known for its compatibility with various model formats, including TensorFlow, PyTorch and ONNX, providing developers with a flexible solution for intregrating and optimizing models from different frameworks. This versatility enables efficient model deployment across diverse hardware and software environments.

## Key Features of TensorRT Models

TensorRT models offer a range of key features that contribute to their efficiency and effectiveness in deep learning inference on NVIDIA GPUs:

- **Precision Calibration:** TensorRT supports precision calibration, allowing models to be fine-tuned for specific accuracy requirements. This includes support for reduced precision formats like INT8 and FP16, which can further boost inference speeds while maintaining competitive accuracy.

- **Layer Fusion:** The TensorRT optimization process includes layer fusion, where multiple layers of a neural network are combined into a single operation. This reduces computational overhead and improves inference speed by minimizing memory access and computation.

- **Dynamic Tensor Memory Management:** TensorRT efficiently manages tensor memory usage during inference, reducing memory overhead and optimizing memory allocation. This results in more efficient GPU utilization.

- **Automatic Kernel Tuning:** TensorRT applies automatic kernel tuning to select the most optimized GPU kernel for each layer of the model. This adaptive approach ensures that the model takes full advantage of the computational power of the GPU.


## Usage

!!! example "Usage"

    === "Python"

        ```python
        
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt", verbose=True)

        model.export(format="engine", device=0, half=True)

        model_engine = Vajra("vajra-v1-nano-det.engine")

        model_engine.val(data="coco.yaml", device=0, img_size=640, half=True)

        result = model_engine.predict("path/to/img.jpg")
        ```
    
    === "CLI"

        ```bash

        vajra export model=vajra-v1-nano-det.pt format=engine device=0 half=True

        vajra val model=vajra-v1-nano-det.engine data=coco.yaml device=0 img_size=640 half=True

        vajra predict model=vajra-v1-nano-det.engine source='path/to/img.jpg'

        ```

## Export Arguments for TensorRT

| Argument    | Type              | Default        | Description                                                                                                                                                                                                                                                      |
| ----------- | ----------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`             | `'engine'`     | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `img_size`     | `int` or `tuple`  | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `half`      | `bool`            | `False`        | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                                                                                     |
| `int8`      | `bool`            | `False`        | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.                                                                    |
| `dynamic`   | `bool`            | `False`        | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `simplify`  | `bool`            | `True`         | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                                                                                                                                                                 |
| `workspace` | `float` or `None` | `None`         | Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance; use `None` for auto-allocation by TensorRT up to device maximum.                                                                                      |
| `nms`       | `bool`            | `False`        | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                                                              |
| `batch`     | `int`             | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`      | `str`             | `'coco8.yaml'` | Path to the dataset configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
| `fraction`  | `float`           | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |
| `device`    | `str`             | `None`         | Specifies the device for exporting: GPU (`device=0`), DLA for NVIDIA Jetson (`device=dla:0` or `device=dla:1`).                                                                                                                                                  |

## Exporting TensorRT with INT8 Quantization

Exporting VajraV1 models using TensorRT with INT8 precision executes post-training quantization (PTQ). TensorRT uses calibration for PTQ, which measures the distribution of activations within each activation tensor as the VajraV1 model processes inference on representative input data, and then uses that distribution to estimate scale values for each tensor. Each activation tensor that is a candidate for quantization has an associated scale that is deduced by a calibration process.

!!! example "Usage"

    === "Python"

        ```python
        
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt", verbose=True)

        model.export(format="engine", dynamic=True, batch=8, workspace=4, int8=True, data="coco.yaml", device=0)

        model_engine = Vajra("vajra-v1-nano-det.engine")

        result = model_engine.predict("path/to/img.jpg")
        ```
    
    === "CLI"

        ```bash

        vajra export model=vajra-v1-nano-det.pt format=engine batch=8 workspace=4 int8=True data=coco.yaml

        vajra predict model=vajra-v1-nano-det.engine source='path/to/img.jpg'

        ```

TensorRT will generate a callibration cache which can be reused to speed up export of future model weights using the same data, but this may result in poor calibration when the data is vastly different or if the `batch` value is changed drastically. In these circumstances, the existing cache should be renamed and moved to a different directory or deleted entirely.

It is **critical** to ensure that the same device that will use the TensorRt model weights for deployment is used for exporting with INT8 precision, as the calibration resilts can vary across devices.

## Configuring INT8 Export

The arguments provided when using [export](../modes/export.md) for a VajraV1 model will **greatly** influence the performance of the exported model. They will also need to be selected based on the device resources available, however the default arguments _should_ work for most [Ampere (or newer) NVIDIA discrete GPUs](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/). The calibration algorithm used is "MINIMAX_CALIBRATION" and you can read more details about the options available [in the TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/MinMaxCalibrator.html). 

- `workspace` : Controls the size in (GiB) of the device memory allocation while converting the model weights.
    - Adjust the `workspace` value according to your calibration needs and resource availability. While a larger `workspace` may increase calibration time, it allows TensorRT to explore a wider range of optimization tactics, potentially enhancing model performance and accuracy. Conversely, a smaller `workspace` can reduce calibration time but may limit the optimization strategies, affecting the quality of the quantized model.

    - Default is `workspace=None`, which will allow for TensorRT to automatically allocate memory, when configuring manually, this value may need to be increased if calibration crashes (exits without warning).

    - TensorRT will report `UNSUPPORTED_STATE` during export if the value for `workspace` is larger than the memory available to the device, which means the value for `workspace` should be lowered or set to `None`.

    - If `workspace` is set to max value and calibration fails/crashes, consider using `None` for auto-allocation or by reducing the values for `img_size` and `batch` to reduce memory requirements.

    - <u><b>Remember</b> calibration for INT8 is specific to each device</u>, borrowing a "high-end" GPU for calibration, might result in poor performance when inference is run on another device.

- `batch` : The maximum batch-size that will be used for inference. During inference smaller batches can be used, but inference will not accept batches any larger than what is specified.

Experimentation by NVIDIA led them to recommend using at least 500 calibration images thaat are representative of the data for your model, with INT8 quantization calibration. This is a guideline and not a hard requirement, and <u>**you will need to experiment with what is required to perform well for your dataset**.</u> Since the calibration data is required for INT8 calibration with TensorRT, make certain to use the `data` argument when using `int8=True` for TensorRT and use `data="my_dataset.yaml"`, which will use the images from validation to calibrate with. When no value is passed for `data` with export to TensorRT with INT8 quantization, the default will be to use one of the small datasets based on the model task instead of throwing an error.

## Advantages of using VajraV1 with TensorRT INT8

- **Reduced model size:** Quantization from FP32 to INT8 can reduce the model size by 4x (on disk or in memory), leading to faster download times. lower storage requirements, and reduced memory footprint when deploying a model.

- **Lower power consumption:** Reduced precision operations for INT8 exported VajraV1 models can consume less power compared to FP32 models, especially for battery-powered devices.

- **Improved inference speeds:** TensorRT optimizes the model for the target hardware, potentially leading to faster inference speeds on GPUs, embedded devices, and accelerators.

??? note "Note on Inference Speeds"

    The first few inference calls with a model exported to TensorRT INT8 can be expected to have longer than usual preprocessing, inference, and/or postprocessing times. This may also occur when changing `img_size` during inference, especially when `img_size` is not the same as what was specified during export (export `img_size` is set as TensorRT "optimal" profile).

## Summary

In this guide, we focused on converting VajraV! models to NVIDIA's TensorRT model format. This conversion step is crucial for improving the efficiency and speed of VajraV1 models, making them more effective and suitable for diverse deployment environments.

For more information on usage details, take a look at the [TensorRT official documentation](https://docs.nvidia.com/deeplearning/tensorrt/).

If you're curious about additional VajraV1 integrations, our [integration guide page](../integrations/index.md) provides an extensive selection of informative resources and insights.
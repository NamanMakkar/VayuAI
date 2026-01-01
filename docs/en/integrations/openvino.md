# Intel OpenVINO Export

In this guide we will cover exorting VajraV1 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating VajraV1 inference on Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.

OpenVINO, short for Open Visual Inference & Neural Network Optimization toolkit is a comprehensive toolkit for optimizing and deploying AI inference models. Even thought the name contains Visual, OpenVINO also supports various additional tasks including language audio, time series etc.

## Usage Examples

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")

        model.export(format="openvino")

        ov_model = Vajra("vajra-v1-nano-det_openvino_model/")

        # Run inference on CPU
        results=  ov_model("path/to/img.jpg")

        # Run inference with specific device: ["intel:gpu", "intel:cpu", "intel:npu"]
        results = ov_model("path/to/img.jpg", device="intel:gpu")
        ```

    === "CLI"

        ```bash
        # Export the VajraV1 PyTorch model to OpenVINO format
        vajra export model=vajra-v1-nano-det.pt format=openvino

        # Run inference on CPU
        vajra predict model=vajra-v1-nano-det_openvino_model source="path/to/img.jpg"
        
        # Run inference with specific device: ["intel:gpu", "intel:cpu", "intel:npu"]
        vajra predict model=vajra-v1-nano-det_openvino_model source="path/to/img.jpg" device="intel:gpu"
        ```

## Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'openvino'`   | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `img_size`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `half`     | `bool`           | `False`        | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                                                                                     |
| `int8`     | `bool`           | `False`        | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.                                                                    |
| `dynamic`  | `bool`           | `False`        | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `nms`      | `bool`           | `False`        | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                                                              |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | Path to the dataset configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |

For more details about the export process, visit the [documentation page on exporting](../modes/export.md).

!!! warning

    OpenVINO™ is compatible with most Intel® processors but to ensure optimal performance:

    1. Verify OpenVINO™ support
        Check whether your Intel® chip is officially supported by OpenVINO™ using [Intel's compatibility list](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).

    2. Identify your accelerator
        Determine if your processor includes an integrated NPU (Neural Processing Unit) or GPU (integrated GPU) by consulting [Intel's hardware guide](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html).

    3. Install the latest drivers
        If your chip supports an NPU or GPU but OpenVINO™ isn't detecting it, you may need to install or update the associated drivers. Follow the [driver‑installation instructions](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98) to enable full acceleration.

    By following these three steps, you can ensure OpenVINO™ runs optimally on your Intel® hardware.

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated and discrete GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular deep learning frameworks such as PyTorch, TensorFlow, TensorFlow Lite, Keras, ONNX, PaddlePaddle, and Caffe.
4. **Ease of Use**: The toolkit comes with more than [80 tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) teaching different aspects of the toolkit.

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once your model is successfully exported to the OpenVINO format, you have two primary options for running inference:

1. Use the `VayuAI` package, which provides a high-level API and wraps the OpenVINO Runtime.

2. Use the native `openvino` package for more advanced or customized control over inference behavior.

### Inference with VayuAI

```python

from vajra import Vajra

ov_model = Vajra("vajra-v1-nano-det_openvino_model/")
ov_model.predict(device="intel:gpu")

```

This approach is ideal for fast prototyping or deployment when you don't need full control over the inference pipeline.

### Inference with OpenVINO Runtime

The openvino Runtime provides a unified API to inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution.

You'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

## FAQ

### How do I export VajraV1 models to OpenVINO format?

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")

        # Export the VajraV1 PyTorch model to OpenVINO format
        model.export(format="openvino") # creates "vajra-v1-nano-det_openvino_model/"
        ```

    === "CLI"

        ```bash
        # Export the VajraV1 PyTorch model to OpenVINO format
        vajra export model=vajra-v1-nano-det.pt format=openvino # creates "vajra-v1-nano-det_openvino_model/"
        ```

For more information, refer to the [export formats documentation](../modes/export.md).

### What are the benefits of using OpenVINO with VajraV1 models?

Using Intel's OpenVINO toolkit with VajraV1 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

### How can I run inference using a VajraV1 model exported to OpenVINO?

After exporting a VajraV1 model to OpenVINO format, you can run inference using Python or CLI:

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        ov_model = Vajra("vajra-v1-nano-det.pt")

        result = ov_model("path/to/img.jpg")
        ```

    === "CLI"

        ```bash
        vajra predict model=vajra-v1-nano-det_openvino_model source="path/to/img.jpg"
        ```

Refer to the [predict mode documentation](../modes//predict.md) for more details.







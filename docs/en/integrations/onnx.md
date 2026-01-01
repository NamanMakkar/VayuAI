# ONNX Export for VajraV1 Models

Often, when deploying computer vision models, you will need a model format that is both flexible and compatible with multiple platforms.

Exporting VajraV1 to ONNX format streamlines deployment and ensures optimal performance across various platforms.

## Key features of ONNX

The ability of ONNX to handle various formats can be attributed to the following key features:

- **Common Model Representation**: ONNX defines a common set of operators (like convolutions, layers, etc.) and a standard data format. When a model is converted to ONNX format, its architecture and weights are translated into this common representation. This uniformity ensures that the model can be understood by any framework that supports ONNX.

- **Versioning and Backward Compatibility**: ONNX maintains a versioning system for its operators. This ensures that even as the standard evolves, models created in older versions remain usable. Backward compatibility is a crucial feature that prevents models from becoming obsolete quickly.

- **Graph-based Model Representation**: ONNX represents models as computational graphs. This graph-based structure is a universal way of representing machine learning models, where nodes represent operations or computations, and edges represent the tensors flowing between them. This format is easily adaptable to various frameworks which also represent models as graphs.

- **Tools and Ecosystem**: There is a rich ecosystem of tools around ONNX that assist in model conversion, visualization, and optimization. These tools make it easier for developers to work with ONNX models and to convert models between different frameworks seamlessly.

## Common Usage of ONNX Models

### CPU Deployment

ONNX models are often deployed on CPUs due to their compatibility with ONNX Runtime. This runtime is optimized for CPU execution. It significantly improves inference speed and makes real-time CPU deployments feasible.

### Supported Deployment Options

While ONNX models are commonly used on CPUs, they can also be deployed on the following platforms:

- **GPU Acceleration**: ONNX fully supports GPU acceleration, particularly NVIDIA CUDA. This enables efficient execution on NVIDIA GPUs for tasks that demand high computational power.

- **Edge and Mobile Devices**: ONNX extends to edge and mobile devices, perfect for on-device and real-time inference scenarios. It's lightweight and compatible with edge hardware.

- **Web Browsers**: ONNX can run directly in web browsers, powering interactive and dynamic web-based AI applications.

## Exporting VajraV1 Models to ONNX

You can expand model compatibility and deployment flexibility by converting VajraV1 models to ONNX format. VajraV1 provides a straightforward export process that can significantly enhance your model's performance across different platforms.

### Installation

To install run:

```bash
git clone https://github.com/NamanMakkar/VayuAI.git
cd VayuAI/
pip install .
```

### Usage Example


!!! example "Usage"

    === "Python"

        ```python
        
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt", verbose=True)

        model.export(format="onnx", device=0)

        model_onnx = Vajra("vajra-v1-nano-det.onnx")

        model_onnx.val(data="coco.yaml", device=0, img_size=640)
        ```
    
    === "CLI"

        ```bash

        vajra export model=vajra-v1-nano-det.pt format=onnx device=0

        vajra val model=vajra-v1-nano-det.onnx data=coco.yaml device=0 img_size=640

        ```

## Export Arguments for ONNX


When exporting your YOLO11 model to ONNX format, you can customize the process using various arguments to optimize for your specific deployment needs:

| Argument   | Type             | Default  | Description                                                                                                                                 |
| ---------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'onnx'` | Target format for the exported model, defining compatibility with various deployment environments.                                          |
| `img_size`    | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.           |
| `half`     | `bool`           | `False`  | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                |
| `dynamic`  | `bool`           | `False`  | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                     |
| `simplify` | `bool`           | `True`   | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                                            |
| `opset`    | `int`            | `None`   | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version. |
| `nms`      | `bool`           | `False`  | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                         |
| `batch`    | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.     |
| `device`   | `str`            | `None`   | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                             |

## Deploying Exported VajraV1 Models

Once you've successfully exported your VajraV1 models to ONNX format, the next step is deploying these models in various environments. For detailed instructions on deploying your ONNX models, take a look at the following resources:

- **[ONNX Runtime Python API Documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)**: This guide provides essential information for loading and running ONNX models using ONNX Runtime.

- **[Deploying on Edge Devices](https://onnxruntime.ai/docs/tutorials/iot-edge/)**: Check out this docs page for different examples of deploying ONNX models on edge.

- **[ONNX Tutorials on GitHub](https://github.com/onnx/tutorials)**: A collection of comprehensive tutorials that cover various aspects of using and implementing ONNX models in different scenarios.

- **[Triton Inference Server](../guides/triton-inference-server.md)**: Learn how to deploy your ONNX models with NVIDIA's Triton Inference Server for high-performance, scalable deployments.

## Summary

This guide teaches you how to export Vayuvahana Technologies VayuAI's VajraV1 models to the ONNX format to increase their interoperability and performance across various platforms. You were also introduced to the ONNX Runtime and ONNX deployment options.

ONNX export is just one of many [export formats](../modes/export.md) supported by VajraV1, allowing you to deploy your models in virtually any environment. Depending on your specific needs, you might also want to explore other export options like [TensorRT](../integrations/tensorrt.md) for maximum GPU performance or [CoreML](../integrations/coreml.md) for Apple devices.

For further details on usage, visit the [ONNX official documentation](https://onnx.ai/onnx/intro/).
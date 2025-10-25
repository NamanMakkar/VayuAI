# Model Export with Vayuvahana Technologies VayuAI

## Introduction

Export Mode in Vayuvahana Technologies VayuAI offers a versatile range of options for exporting trained models to different formats, making it deployable across various platforms and devices. This guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

## Why Choose Vayuvahana Technologies VayuAI's Export Mode

- **Versatility:** Export to multiple formats including ONNX, TensorRT, CoreML, and more.
- **Performance:** Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or OpenVINO.
- **Compatibility:** Make your model universally deployable across numerous hardware and software environments.
- **Ease of Use:** Simple CLI and Python API for quick and straightforward model exporting.

### Key Features of Export Mode

Here are some of the standout functionalities:

- **One-Click Export:** Simple commands for exporting to different formats.
- **Batch Export:** Export batched-inference capable models.
- **Optimized Inference:** Exported models are optimized for quicker inference times.
- **Tutorial Videos:** In-depth guides and tutorials for a smooth exporting experience.

!!! tip

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Export a VajraV1-nano-det model to a different format like ONNX or TensorRT. See the Arguments section below for a full list of export arguments.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Load a model
        model = Vajra("vajra-v1-nano-det.pt")  # load an official model
        model = Vajra("path/to/best-vajra-v1-nano-det.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        vajra export model=vajra-v1-nano-det.pt format=onnx      # export official model
        vajra export model=path/to/best-vajra-v1-nano-det.pt format=onnx # export custom trained model
        ```

## Arguments

This table details the configurations and options available for exporting VajraV1 models to different formats. These settings are critical for optimizing the exported model's performance, size and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

{% include "macros/export-args.md" %}

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and accuracy.

## Export Formats

Available VajraV1 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `vajra predict model=vajra-v1-nano-det.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

## FAQ

### How do I export a VajraV1 model to ONNX format?

Exporting a VajraV1 model to ONNX format is straightforward with the VayuAI SDK. It provides both Python and CLI methods for exporting models.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Load a model
        model = Vajra("vajra-v1-nano-det.pt")  # load an official model
        model = YOLO("path/to/best-vajra-v1-nano-det.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        vajra export model=vajra-v1-nano-det.pt format=onnx      # export official model
        vajra export model=path/to/best-vajra-v1-nano-det.pt format=onnx # export custom trained model
        ```

### What are the benefits of using TensorRT for model export?

Using TensorRT for model export offers significant performance improvements. VajraV1 models exported to TensorRT can achieve upto a 5x GPU speedup, making it ideal for real-time inference applications.

- **Versatility:** Optimize models for a specific hardware setup.
- **Speed:** Achieve faster inference through advanced optimizations.
- **Compatibility:** Integrate smoothly with NVIDIA hardware.

### How do I enable INT8 quantization when exporting my VajraV1 model?

INT8 quantization is an excellent way to compress the model and speed up inference, especially on edge devices. Here is how you can enable INT8 quantization:

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")  # Load a model
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        vajra export model=vajra-v1-nano-det.pt format=engine int8=True # export TensorRT model with INT8 quantization
        ```

INT8 quantization can be applied to various formats, such as TensorRT, OpenVINO and CoreML. For optimal quantization results, provide a representative dataset using the `data` parameter.

### Why is dynamic input size important when exporting models?

Dynamic input size allows the exported model to handle varying image dimensions, providing flexibility and optimizing processing efficiency for different use cases. When exporting to formats like ONNX, or TensorRT, enabling dynamic input size ensures that the model can adapt to different input shapes seamlessly.

To enable this feature, use the `dynamic=True` flag during export:

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        vajra export model=vajra-v1-nano-det.pt format=onnx dynamic=True
        ```

Dynamic input sizing is particularly useful for applications where input dimensions may vary, such as video processing or when handling images from different sources.

### What are the key export arguments to consider for optimizing model performance?

Understanding and configuring export arguments is crucial for optimizing model performance:

- **`format:`** The target format for the exported model (e.g., `onnx`, `engine`, `openvino`, `torchscript`, `tensorflow`).
- **`img_size:`** Desired image size for the model input (e.g., `640` or `(height, width)`).
- **`half:`** Enables FP16 quantization, reducing model size and potentially speeding up inference.
- **`optimize:`** Applies specific optimizations for mobile or constrained environments.
- **`int8:`** Enables INT8 quantization, highly beneficial for edge AI deployments.

For deployment on specific hardware platforms, consider using specialized export formats like TensorRT for NVIDIA GPUs, CoreML for Apple devices, or Edge TPU for Google Coral devices.

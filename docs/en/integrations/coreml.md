# CoreML Export for VajraV1 Models

Deploying computer vision models on Apple devices like iPhones and Macs requires a format that ensures seamless performance.

The CoreML export format allows you to optimize your VajraV1 models for efficient object detection in iOS and macOS applications. In this guide, we'll walk you through the steps for converting your models to the CoreML format, making it easier for your models to perform well on Apple devices.

Applications can take advantage of Core ML without the need to have a network connection or API calls because the Core ML framework works using on-device computing. This means model inference can be performed locally on the user's device.

## Key Features of CoreML Models

Apple's CoreML framework offers robust features for on-device machine learning. Here are the key features that make CoreML a powerful tool for developers:

- **Comprehensive Model Support:** Converts and runs models from popular frameworks like TensorFlow, PyTorch, scikit-learn, XGBoost, and LibSVM.

- **On-device Machine Learning**: Ensures data privacy and swift processing by executing models directly on the user's device, eliminating the need for network connectivity.

- **Performance and Optimization**: Uses the device's CPU, GPU, and Neural Engine for optimal performance with minimal power and memory usage. Offers tools for model compression and optimization while maintaining accuracy.

- **Ease of Integration**: Provides a unified format for various model types and a user-friendly API for seamless integration into apps. Supports domain-specific tasks through frameworks like Vision and Natural Language.

- **Advanced Features**: Includes on-device training capabilities for personalized experiences, asynchronous predictions for interactive ML experiences, and model inspection and validation tools.

## CoreML Deployment Options

Before we look at the code for exporting VajraV1 models to the CoreML format, let's understand where CoreML models are usually used.

CoreML offers various deployment options for machine learning models, including:

- **On-Device Deployment**: This method directly integrates CoreML models into your iOS app. It's particularly advantageous for ensuring low latency, enhanced privacy (since data remains on the device), and offline functionality. This approach, however, may be limited by the device's hardware capabilities, especially for larger and more complex models. On-device deployment can be executed in the following two ways.
    - **Embedded Models**: These models are included in the app bundle and are immediately accessible. They are ideal for small models that do not require frequent updates.

    - **Downloaded Models**: These models are fetched from a server as needed. This approach is suitable for larger models or those needing regular updates. It helps keep the app bundle size smaller.

- **Cloud-Based Deployment**: CoreML models are hosted on servers and accessed by the iOS app through API requests. This scalable and flexible option enables easy model updates without app revisions. It's ideal for complex models or large-scale apps requiring regular updates. However, it does require an internet connection and may pose latency and security issues.

## Exporting VajraV1 Models to CoreML

Exporting VajraV1 to CoreML enables optimized, on-device machine learning performance within Apple's ecosystem, offering benefits in terms of efficiency, security, and seamless integration with iOS, macOS, watchOS, and tvOS platforms.

## Usage Example

!!! example "Usage"

    === "Python"

        ```python

        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")

        model.export(format="coreml")

        coreml_model = Vajra("vajra-v1-nano-det.mlpackage")

        results = coreml_model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        vajra export model=vajra-v1-nano-det.pt format=coreml

        vajra predict model=vajra-v1-nano-det.mlpackage source="path/to/img.jpg"
        ```

## Export Arguments

| Argument | Type             | Default    | Description                                                                                                                                                                                   |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `img_size`  | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `half`   | `bool`           | `False`    | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                  |
| `int8`   | `bool`           | `False`    | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices. |
| `nms`    | `bool`           | `False`    | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`  | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `device` | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                               |

## Deploying Exported VajraV1 CoreML Models

Having successfully exported your VajraV1 models to CoreML, the next critical phase is deploying these models effectively. For detailed guidance on deploying CoreML models in various environments, check out these resources:

- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: This guide includes instructions and examples to convert models from TensorFlow, PyTorch, and other libraries to Core ML.

- **[ML and Vision](https://developer.apple.com/videos/)**: A collection of comprehensive videos that cover various aspects of using and implementing CoreML models.

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: A comprehensive guide on integrating a CoreML model into an iOS application, detailing steps from preparing the model to implementing it in the app for various functionalities.

## Summary

In this guide, we went over how to export Vayuvahana Technologies VayuAI's VajraV1 models to COREML format. By following the steps outlined in this guide, you can ensure maximum compatibility and performance when exporting VajraV1 models to CoreML.

For further details on usage, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

For more details on other integrations, visit the [integrations guide page](../integrations/index.md). 

## FAQ

### How do I export VajraV1 models to CoreML format?

To export your VajraV1 models to CoreML format, you'll first need to ensure you have the VayuAI SDK installed.

!!! example "Installation"

    === "CLI"

        ```bash
        git clone https://github.com/NamanMakkar/VayuAI.git
        cd VayuAI/
        pip install .
        ```

Next, you can export the model using the following Python or CLI commands:

!!! example "Usage"

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")
        model.export(format="coreml")
        ```

    === "CLI"

        ```bash
        vajra export model=vajra-v1-nano-det.pt format=coreml
        ```

For further details, refer to the [Exporting VajraV1 Models to CoreML](../modes/export.md) section of our documentation.

### What are the benefits of using CoreML for deploying VajraV1 models?

CoreML provides numerous advantages for deploying [Vayuvahana Technologies VayuAI's VajraV1](https://github.com/NamanMakkar/VayuAI) models on Apple devices:

- **On-device Processing**: Enables local model inference on devices, ensuring data privacy and minimizing latency.
- **Performance Optimization**: Leverages the full potential of the device's CPU, GPU, and Neural Engine, optimizing both speed and efficiency.
- **Ease of Integration**: Offers a seamless integration experience with Apple's ecosystems, including iOS, macOS, watchOS, and tvOS.
- **Versatility**: Supports a wide range of machine learning tasks such as image analysis, audio processing, and natural language processing using the CoreML framework.

For more details on integrating your CoreML model into an iOS app, check out the guide on [Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app).

### What are the deployment options for VajraV1 models exported to CoreML?

Once you export your VajraV1 model to CoreML format, you have multiple deployment options:

1. **On-Device Deployment**: Directly integrate CoreML models into your app for enhanced privacy and offline functionality. This can be done as:
    - **Embedded Models**: Included in the app bundle, accessible immediately.
    - **Downloaded Models**: Fetched from a server as needed, keeping the app bundle size smaller.

2. **Cloud-Based Deployment**: Host CoreML models on servers and access them via API requests. This approach supports easier updates and can handle more complex models.

For detailed guidance on deploying CoreML models, refer to [CoreML Deployment Options](#coreml-deployment-options).

### How does CoreML ensure optimized performance for VajraV1 models?

CoreML ensures optimized performance for [Vayuvahana Technologies VayuAI's VajraV1](https://github.com/NamanMakkar/VayuAI) models by utilizing various optimization techniques:

- **Hardware Acceleration**: Uses the device's CPU, GPU, and Neural Engine for efficient computation.
- **Model Compression**: Provides tools for compressing models to reduce their footprint without compromising accuracy.
- **Adaptive Inference**: Adjusts inference based on the device's capabilities to maintain a balance between speed and performance.

For more information on performance optimization, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

### Can I run inference directly with the exported CoreML model?

Yes, you can run inference directly using the exported CoreML model. Below are the commands for Python and CLI:

!!! example "Running Inference"

    === "Python"

        ```python
        from vajra import Vajra

        coreml_model = Vajra("vajra-v1-nano-det.mlpackage")
        results = coreml_model("path/to/img.jpg")
        ```

    === "CLI"

        ```bash
        vajra predict model=vajra-v1-nano-det.mlpackage source='path/to/img.jpg'
        ```

Refer to the [Usage Example](#usage-example) of the CoreML export guide.
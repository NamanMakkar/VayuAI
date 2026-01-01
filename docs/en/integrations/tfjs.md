# Export VajraV1 to a TF.js Model Format

Deploying machine learning models directly in the browser or on Node.js can be tricky. You'll need to make sure your model format is optimized for faster performance so that the model can be used to run interactive applications locally on the user's device. The TensorFlow.js, or TF.js, model format is designed to use minimal power while delivering fast performance.

## Key Features of TF.js

- **Cross-Platform Support:** TensorFlow.js can be used in both browser and Node.js environments, providing flexibility in deployment across different platforms. It lets developers build and deploy applications more easily.

- **Support for Multiple Backends:** TensorFlow.js supports various backends for computation including CPU, WebGL for GPU acceleration, WebAssembly (WASM) for near-native execution speed, and WebGPU for advanced browser-based machine learning capabilities.

- **Offline Capabilities:** With TensorFlow.js, models can run in the browser without the need for an internet connection, making it possible to develop applications that are functional offline.

## Deployment Options with TF.js

- **In-Browser ML Applications:** You can build web applications that run machine learning models directly in the browser. The need for server-side computation is eliminated and the server load is reduced.

- **Node.js Applications:** TensorFlow.js also supports deployment in Node.js environments, enabling the development of server-side machine learning applications. It is particularly useful for applications that require the processing power of a server or access to a server side data.

- **Chrome Extensions:** An interesting deployment scenario is the creation of Chrome extensions with TensorFlow.js. For instance, you can develop an extension that allows users to right-click on an image within any webpage to classify it using a pre-trained ML model. TensorFlow.js can be integrated into everyday web browsing experiences to provide immediate insights or augmentations based on machine learning.

## Usage


```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.export(format="tfjs")
tfjs_model = Vajra("./vajra-v1-nano-det_web_model")

results = tfjs_model("/path/to/img.jpg")
```

```bash
vajra export model=vajra-v1-nano-det.pt format=tfjs # creates /vajra-v1-nano-det_web_model
vajra predict model="./vajra-v1-nano-det_web_model" source="/path/to/img.jpg"
```

## Export Arguments

| Argument | Type             | Default  | Description                                                                                                                                                                                   |
| -------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'tfjs'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `img_size`  | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `half`   | `bool`           | `False`  | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                  |
| `int8`   | `bool`           | `False`  | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices. |
| `nms`    | `bool`           | `False`  | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`  | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `device` | `str`            | `None`   | Specifies the device for exporting: CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                                                 |

For more details about the export process, visit the [documentation page on exporting](../modes/export.md).

## Deploying Exported VajraV1 TensorFlow.js Models

The recommended first step for running a TF.js model is to use Vajra("vajra-v1-nano-det_web_model") method as preiously shown in the Usage code snippet.

For in-depth instructions on deployment take a look at the following:

- **[Chrome Extension](https://www.tensorflow.org/js/tutorials/deployment/web_ml_in_chrome)**: Here's the developer documentation for how to deploy your TF.js models to a Chrome extension.

- **[Run TensorFlow.js in Node.js](https://www.tensorflow.org/js/guide/nodejs)**: A TensorFlow blog post on running TensorFlow.js in Node.js directly.

- **[Deploying TensorFlow.js - Node Project on Cloud Platform](https://www.tensorflow.org/js/guide/node_in_cloud)**: A TensorFlow blog post on deploying a TensorFlow.js model on a Cloud Platform.

## Summary

In this guide we learnt how to export the VajraV1 models to the TF.js format. By exporting to TF.js, you gain the flexibility to optimize, deploy and scale your VajraV1 models on a wide range of platforms.

For further details, visit the [TF.js official documentation](https://www.tensorflow.org/js/guide).

## FAQ

### How do I export VajraV1 Models to TensorFlow.js format?

Exporting VajraV1 models to TF.js format is straightforward. You can follow these steps:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.export(format="tfjs")
tfjs_model = Vajra("./vajra-v1-nano-det_web_model")

results = tfjs_model("/path/to/img.jpg")
```

```bash
vajra export model=vajra-v1-nano-det.pt format=tfjs # creates /vajra-v1-nano-det_web_model
vajra predict model="./vajra-v1-nano-det_web_model" source="/path/to/img.jpg"
```

### Why should I export my VajraV1 model to TensorFlow.js?

Exporting VajraV1 models to TF.js offers the following advantages:

1. **Local Execution:** Models can run directly in the browser or Node.js, reducing latency and enhancing user experience.
2. **Cross-Platform Support:** TF.js supports multiple environments, allowing flexibility in deployment.
3. **Offline Capabilities:** Enables applications to function without an internet connection, ensuring reliability and privacy.
4. **GPU Acceleration:** Leverages WebGL for GPU acceleration, optimizing performance on devices with limited resources.

### How does TensorFlow.js benefit browser-based machine learning applications?

TensorFlow.js is specifically designed for efficient execution of ML models in browsers and Node.js environments. Here's how it benefits browser-based applications:

- **Reduces Latency:** Runs machine learning models locally, providing immediate results without relying on server-side computations.
- **Improves Privacy:** Keeps sensitive data on the user's device, minimizing security risks.
- **Enables Offline Use:** Models can operate without an internet connection, ensuring consistent functionality.
- **Supports Multiple Backends:** Offers flexibility with backends like CPU, WebGL, WebAssembly (WASM), and WebGPU for varying computational needs.

Check out the [official TF.js guide](https://www.tensorflow.org/js/guide).

### Can I deploy a VajraV1 model on server-side Node.js applications using TensorFlow.js?

Yes, TensorFlow.js allows the deployment of VajraV1 models on Node.js environments. This enables server-side ML applications that benefit from the processing power of a server and access to server-side data.

To get started with Node.js deployment, refer to the [Run TF.js in Node.js](https://www.tensorflow.org/js/guide/nodejs) guide from TensorFlow.
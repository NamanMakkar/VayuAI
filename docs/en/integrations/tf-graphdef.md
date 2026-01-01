# Export VajraV1 Models to TF GraphDef for Deployment

When deploying cutting-edge Vision AI models like VajraV1 in different environments you might run into compatibility issues. Google's TensorFlow GraphDef or TF GraphDef, offers a solution by providing a serialized, platform-independent representation of your model. Using the TF GraphDef model format, you can deploy VajraV1 models in environments where the complete TensorFlow ecosystem may not be available, such as mobile devices or specialized hardware.

In this guide, you will learn how to export your VajraV1 models to the TF GraphDef model format.

## Why Export to TF GraphDef

TF GraphDef is a component of the TensorFlow ecosystem that was developed by Google. It can be used to optimize and deploy models like VajraV1. Exporting to TF GraphDef lets us move models from research to real-world applications. It allows models to run in environments without the full TensorFlow framework.

The GraphDef format represents the model as a serialized computation graph. This enables various optimization techniques like constant folding, quantization, and graph transformations. These optimizations ensure efficient execution, reduced memory usage, and faster inference speeds.

GraphDef models can use hardware accelerators like GPUs and TPUs. The TF GraphDef format creates a self contained package with the model and its dependencies, simplifying deployment and integration into diverse systems.

## Key Features of TF GraphDef Models

- **Model Serialization:** TF GraphDef provides a way to serialize and store TensorFlow models in a platform-independent format. This serialized representation allows you to load and execute your models without the original Python codebase, making deployment easier.

- **Graph Optimization:** TF GraphDef enables the optimization of computational graphs. These optimizations can boost performance by streamlining execution flow, reducing redundancies, and tailoring operations to suit specific hardware.

- **Deployment Flexibility:** Models exported to the GraphDef format can be used in various environments, including resource-constrained devices, web browsers, and systems with specialised hardware. This opens up possibilities for wider deployment of your TensorFlow models.

- **Production Focus:** GraphDef is designed for production deployment. It supports efficient execution, serialization features, and optimizations that align with real-world use cases.

## Deployment Options with TF GraphDef

- **TensorFlow Serving:** This framework is designed to deploy TensorFlow models in production environments. TensorFlow Serving offers model management, versioning, and the infrastructure for efficient model serving at scale. It is a seamless way to integrate your GraphDef-based models into production web services or APIs/

- **Mobile and Embedded Devices:** With tools like [TensorFlow Lite](../integrations/tflite.md), you can convert TF GraphDef models into formats optimized for smartphones, tablets, and various embedded devices. Your models can then be used for on-device inference, where execution is done locally, often providing performance gains and offline capabilities.

- **Web Browsers:** [TensorFlow.js](../integrations/tfjs.md) enables the deployment of TF GraphDef models directly within web browsers. It paves the way for real-time object detection applications running on the client side, using the capabilities of VajraV1 through JavaScript.

- **Specialized Hardware:** TF GraphDef's platform-agnostic nature allows it to target custom hardware, such as accelerators and TPUs. These devices can provide performance advantages for computationally intensive models.

## Exporting VajraV1 Models to TF GraphDef

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")

model.export(format="pb")

tf_graphdef_model = Vajra("vajra-v1-nano-det.pb")

results = tf_graphdef_model("/path/to/img.jpg")
```

```bash
vajra export model=vajra-v1-nano-det.pt format=pb

vajra predict model='vajra-v1-nano-det.pb' source='/path/to/img.jpg'
```

## Export Arguments

| Argument | Type             | Default | Description                                                                                                                             |
| -------- | ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'pb'`  | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `img_size`  | `int` or `tuple` | `640`   | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `batch`  | `int`            | `1`     | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str`            | `None`  | Specifies the device for exporting: CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                           |

For more details about the export process visit the [documentation page on exporting](../modes/export.md).

## Deploying Exported VajraV1 TF GraphDef Models

Once exported, the next step is to deploy your VajraV1 models. The first step recommended for running a TF GraphDef model is to use the Vajra("vajra-v1-nano-det.pb") method, as previously shown in the code snippet.

For more information on deploying your TF GraphDef models, have a look at the following:

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving):** A guide on TensorFlow Serving that teaches how to deploy and serve machine learning models efficiently in production environments.

- **[TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter):** This page describes how to convert machine learning models into a format optimized for on-device inference with TensorFlow Lite.

- **[TensorFlow.js](https://www.tensorflow.org/js/guide/conversion):** A guide on model conversion that teaches how to convert TensorFlow or Keras models into TensorFlow.js format for use in web applications.

## Summary

In this guide we learnt how to export VajraV1 models to the TF GraphDef format. By doing so, you can flexibly deploy your optimized VajraV1 models in different environments.

For further details on usage, visit the [TF GraphDef official documentation](https://www.tensorflow.org/api_docs/python/tf/Graph).

## FAQ

### How do I export a VajraV1 model to TF GraphDef format?

VajraV1 models can be exported to TensorFlow GraphDef (TF GraphDef) format seamlessly using the code snippets below:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")

model.export(format="pb")

tf_graphdef_model = Vajra("vajra-v1-nano-det.pb")

results = tf_graphdef_model("/path/to/img.jpg")
```

```bash
vajra export model=vajra-v1-nano-det.pt format=pb

vajra predict model='vajra-v1-nano-det.pb' source='/path/to/img.jpg'
```

### What are the benefits of using TF GraphDef for VajraV1 model deployment?

1. **Platform Independence:** TF GraphDef provides a platform-independent format, allowing models to be deployed across various environments including mobile and web browsers.
2. **Optimizations:** The format enables several optimizations, such as constant folding, quantization, and graph transformations, which enhance execution efficiency and reduce memory usage.
3. **Hadware Acceleration:** Models in TF GraphDef format can leverage hardware accelerators like GPUs and TPUs for performance gains.

### How can I deploy a VajraV1 model on specialised hardware using TF GraphDef?

Once you have exported a VajraV1 model to GraphDef format, you can deploy it across various specialized hardware platforms. Typical deployment scenarios include:

- **TensorFlow Serving**: Use TensorFlow Serving for scalable model deployment in production environments. It supports model management and efficient serving.
- **Mobile Devices**: Convert TF GraphDef models to TensorFlow Lite, optimized for mobile and embedded devices, enabling on-device inference.
- **Web Browsers**: Deploy models using TensorFlow.js for client-side inference in web applications.
- **AI Accelerators**: Leverage TPUs and custom AI chips for accelerated inference.

Check the [deployment options](#deploying-exported-vajrav1-tf-graphdef-models) section for detailed information.
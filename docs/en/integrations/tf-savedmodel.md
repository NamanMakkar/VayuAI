# Exporting VajraV1 Models to TF SavedModel Format

TF SavedModel is an open-source machine learning framework used by TensorFlow to load machine-learning models in a consistent way. Exporting VajraV1 models to TF SavedModel format can help you deploy models easily across different platforms and environments.

In this guide we will walk you through converting your models to the TF SavedModel format, simplifying the process of running inferences with your models on different devices.

## Why Should You Export to TF SavedModel?

The TF SavedModel format is part of the TensorFlow ecosystem, it is designed to save and serialize TensorFlow models seamlessly. It encapsulates the complete details of the models like the architecture, weights and even compilation information. This makes it straightforward to share, deploy, and continue training across different environments.

The TF SavedModel is compatible with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), TensorFlow Lite, and TensorFlow.js. This makes it easier to share and deploy models across various platforms, including web and mobile applications. The TF SavedModel format is useful for both research and production.

## Key Features of TF SavedModels

- **Portability:** TF SavedModel provides a language-neutral, recoverable, hermetic serialization format. They enable higher-level systems and tools to produce, consume and transform TensorFlow models. SavedModels can be easily shared and deployed across different platforms and environments.

- **Ease of Deployment:** TF SavedModel bundles the computational graph, trained parameters, and the necessary metadata into a single package. They can be easily loaded and used for inference without requiring the original code that built the model. This makes the deployment of TensorFlow models straightforward and efficient in various production environments.

- **Asset Management**: TF SavedModel supports the inclusion of external assets such as vocabularies, embeddings, or lookup tables. These assets are stored alongside the graph definition and variables, ensuring they are available when the model is loaded. This feature simplifies the management and distribution of models that rely on external resources.

## Deployment Options with TF SavedModel

- **TensorFlow Serving:** TensorFlow Serving is a flexible, high-performance serving system designed for production environments. It natively supports TF SavedModels, making it easy to deploy and serve your models on cloud platforms, on-premises servers, or edge devices.

- **Cloud Platforms:** Major cloud providers like [Google Cloud Platform (GCP)](https://cloud.google.com/vertex-ai), [Amazon Web Services (AWS)](https://aws.amazon.com/sagemaker/), and [Microsoft Azure](https://azure.microsoft.com/en-us/services/machine-learning/) offer services for deploying and running TensorFlow models, including TF SavedModels. These services provide scalable and managed infrastructure, allowing you to deploy and scale your models easily.

- **Mobile and Embedded Devices:** TensorFlow Lite, a lightweight solution for running machine learning models on mobile, embedded, and IoT devices, supports converting TF SavedModels to the TensorFlow Lite format. This allows you to deploy your models on a wide range of devices, from smartphones and tablets to microcontrollers and edge devices.

- **TensorFlow Runtime:** TensorFlow Runtime (`tfrt`) is a high-performance runtime for executing TensorFlow graphs. It provides lower-level APIs for loading and running TF SavedModels in C++ environments. TensorFlow Runtime offers better performance compared to the standard TensorFlow runtime. It is suitable for deployment scenarios that require low-latency inference and tight integration with existing C++ codebases.

## Exporting VajraV1 Models to TF SavedModel

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.export(format="saved_model")

tf_savedmodel_model = Vajra("./vajra-v1-nano-det_saved_model")

results = tf_savedmodel_model("/path/to/img.jpg")
```

```bash
vajra export model=vajra-v1-nano-det.pt format=saved_model

vajra predict model="./vajra-v1-nano-det_saved_model" source="/path/to/img.jpg"
```

## Export Arguments

| Argument | Type             | Default         | Description                                                                                                                                                                                   |
| -------- | ---------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'saved_model'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `img_size`  | `int` or `tuple` | `640`           | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `keras`  | `bool`           | `False`         | Enables export to Keras format, providing compatibility with TensorFlow serving and APIs.                                                                                                     |
| `int8`   | `bool`           | `False`         | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices. |
| `nms`    | `bool`           | `False`         | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`  | `int`            | `1`             | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `device` | `str`            | `None`          | Specifies the device for exporting: CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                                                 |

For more details about the export process, visit the [documentation page on exporting](../modes/export.md).

## Deploying Exported VajraV1 TF SavedModel Models

After exporting your VajraV1 model to the TF SavedModel format, the next step is to deploy it. The recommended step for running a SavedModel model is to use Vajra("vajra-v1-nano-det_saved_model/") method.

For in-depth instructions on deploying your TF SavedModel models, refer to the following resources:

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**: Here's the developer documentation for how to deploy your TF SavedModel models using TensorFlow Serving.

- **[Run a TensorFlow SavedModel in Node.js](https://blog.tensorflow.org/2020/01/run-tensorflow-savedmodel-in-nodejs-directly-without-conversion.html)**: A TensorFlow blog post on running a TensorFlow SavedModel in Node.js directly without conversion.

- **[Deploying on Cloud](https://blog.tensorflow.org/2020/04/how-to-deploy-tensorflow-2-models-on-cloud-ai-platform.html)**: A TensorFlow blog post on deploying a TensorFlow SavedModel model on the Cloud AI Platform.

## Summary

This guide explored how to export VajraV1 models to the TF SavedModel format. By exporting to TF SavedModel, you gain the flexibility to optimize, deploy and scale your VajraV1 models on a wide range of platforms.

For further details on usage, visit the [TF SavedModel official documentation](https://www.tensorflow.org/guide/saved_model).

## FAQ

### How do I export a VajraV1 model to TensorFlow SavedModel format?

Exporting a VajraV1 model to TensorFlow SavedModel format is straightforward. You can use either Python or CLI to achieve this:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.export(format="saved_model")

tf_savedmodel_model = Vajra("./vajra-v1-nano-det_saved_model")

results = tf_savedmodel_model("/path/to/img.jpg")
```

```bash
vajra export model=vajra-v1-nano-det.pt format=saved_model

vajra predict model="./vajra-v1-nano-det_saved_model" source="/path/to/img.jpg"
```

Refer to the [Export documentation](../modes/export.md) for more details.

### Why should I use the TensorFlow SavedModel format?

The TensorFlow SavedModel format offers several advantages for model deployment.

- **Portability:** It provides a language-neutral format, making it easy to share and deploy models across different environments.
- **Compatibility:** Integrates seamlessly with tools like TensorFlow Serving, TensorFlow Lite, and TensorFlow.js, which are essential for deploying models on various platforms, including web and mobile applications.
- **Complete encapsulation:** Encodes the model architecture, weights, and compilation information, allowing for straightforward sharing and training continuation.

For more details, check out the [model deployment options](../guides/model-deployment-options.md).

### What are the typical deployment scenarios for TF SavedModel?

TF SavedModel can be deployed in various environments, including:

- **TensorFlow Serving:** Ideal for production environments requiring scalable and high-performance model serving.
- **Cloud Platforms:** Supports major cloud services like Google Cloud Platform (GCP), Amazon Web Services (AWS), and Microsoft Azure for scalable model deployment.
- **Mobile and Embedded Devices:** Using TensorFlow Lite to convert TF SavedModels allows for deployment on mobile devices, IoT devices, and microcontrollers.
- **TensorFlow Runtime:** For C++ environments needing low-latency inference with better performance.

For detailed deployment options, visit the official guides on [deploying TensorFlow models](https://www.tensorflow.org/tfx/guide/serving).

### What are the key features of the TensorFlow SavedModel format?

TF SavedModel format is beneficial for AI developers due to the following features:

- **Portability:** Allows sharing and deployment across various environments effortlessly.
- **Ease of Deployment:** Encapsulates the computational graph, trained parameters, and metadata into a single package, which simplifies loading and inference.
- **Asset Management:** Supports external assets like vocabularies, ensuring they are available when the model loads.

For further details, explore the [official TensorFlow documentation](https://www.tensorflow.org/guide/saved_model).
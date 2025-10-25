# Vayuvahana Technologies VayuAI Modes

## Introduction

Vayuvahana Technologies VayuAI is a versatile framework designed to cover the entire lifecycle of machine learning models from data ingesting and model training to validation, deployment and real-world tracking. Each mode serves a specific purpose and is engineered to offer you the flexibility and efficiency required for different tasks and use-cases.

### Modes at a Glance

Understanding the different **modes** that Vayuvahana Technologies VayuAI supports is critical to getting the most out of your models:

- **Train** mode: Fine-tune your model on custom or preloaded datasets.
- **Val** mode: A post-training checkpoint to validate model performance.
- **Predict** mode: Unleash the predictive power of your model on real-world data.
- **Export** mode: Make your model deployment-ready in various formats.
- **Track** mode: Extend your object detection model into real-time tracking applications.
- **Benchmark** mode: Analyze the speed and accuracy of your model in diverse deployment environments.

This comprehensive guide aims to give you an overview and practical insights into each mode, helping you harness the full potential of VajraV1.

## [Train](train.md)

Train mode is used for training a VajraV1 model on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image. Training is essential for creating models that can recognize specific objects relevant to your application.

[Train Examples](train.md){ .md-button }

## [Val](val.md)

Val mode is used for validating a VajraV1 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its accuracy and generalization performance. Validation helps identify potential issues like overfitting and provides metrics such as mean Average Precision (mAP) to quantify model performance. This mode is crucial for tuning hyperparameters and improving overall model effectiveness.

[Val Examples](val.md){ .md-button }

## [Predict](predict.md)

Predict mode is used for making predictions using a trained VajraV1 model on new images or videos. In this mode, the model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model identifies and localizes objects in the input media, making it ready for real world applications.

[Predict Examples](predict.md){ .md-button }

## [Export](export.md)

Export mode is used for converting a VajraV1 model to formats suitable for deployment across different platforms and devices. This mode transforms your PyTorch model into optimized formats like ONNX, TensorRT, or CoreML, enabling deployment in production environments. Exporting is essential for integrating your model with various software applications or hardware devices, often resulting in significant performance improvements.

[Export Examples](export.md){ .md-button }

## [Track](track.md)

Track mode extends VajraV1's object detection capabilities to track objects across video frames or live streams. This mode is particularly valuable for applications requiring persistent object identification, such as surveillance systems or self-driving cars. Track mode implements sophisticated algorithms like ByteTrack to maintain object identity across frames, even when objects temporarily disappear from view.

[Track Examples](track.md){ .md-button }

## [Benchmark](benchmark.md)

Benchmark mode profiles the speed and accuracy of various export formats for VajraV1. This mode provides comprehensive metrics on model size, accuracy (mAP50-95 for detection tasks or accuracy_top5 for classification), and inference time across different formats like ONNX, OpenVINO, and TensorRT. Benchmarking helps you select the optimal export format based on your specific requirements for speed and accuracy in your deployment environment.

[Benchmark Examples](benchmark.md){ .md-button }


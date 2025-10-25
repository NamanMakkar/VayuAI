# Model Training with Vayuvahana Technologies VajraV1

## Introduction

Train mode in Vayuvahana Technologies VajraV1 is engineered for effective and efficient training of object detection models, instance segmentation models, pose estimation models, image classification models and oriented bounding box object detection models utilizing modern hardware capabilities. This guide covers all the details required to get started with training your own models using VajraV1's features.


## Vayuavahana VajraV1 Training features:

- **Efficiency:** Make the most out of your hardware, whether you're on a single GPU setup or scaling across multiple GPUs.
- **Versatility:** Train on custom datasets in addition to readily available ones like COCO, VOC, VisDrone, DOTAv1, and ImageNet
- **User-Friendly:** Simple yet powerful CLI and Python interfaces for a straightforward training experience.
- **Hyperparameter Flexibility:** A broad range of customizable hyperparameters to fine-tune model performance.
- **Model Diversity:** Train both VajraV1 (from the YOLO family) as well as DFINE and DEIM (from the DETR family) object detectors as well as various image classifiers like the EfficientNet and MobileNet.

### Key Features of the Train Mode:

- **Automatic Dataset Download:** Standard datasets like COCO, VisDrone, VOC, ImageNet are downloaded automatically on first use.
- **Multi-GPU Support:** Scale your training efforts seamlessly across multiple GPUs to expedite the process.
- **Hyperparameter Configuration:** The option to modify the hyperparameters through YAML configuration files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics and visualization of the learning process for better insights.

## Usage Examples

Train VajraV1-nano on the COCO8 dataset for 100 epochs at image size 640. The training device can be specified using the 'device' argument. If no argument is passed GPU 'device=0' will be used if available, otherwise `device='cpu'` will be used. See arguments section below for a full list of training arguments.

!!! warning "Windows Multi-Processing Error"

    On Windows, you may receive a `RuntimeError` when launching the training as a script. Add a `if __name__ == "__main__":` block before your training code to resolve it.

Example "Single-GPU and CPU Training Example"

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det", verbose=True)
model = Vajra("vajra-v1-nano-det.pt")
model = Vajra("vajra-v1-nano-det").load("vajra-v1-nano-det.pt")

results = model.train(data="coco8.yaml", epochs=100, img_size=640)
```

```bash
vajra detect train data=coco8.yaml model=vajra-v1-nano-det epochs=100 img_size=640

vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640
```

### Multi GPU Training

Multi-GPU training allows for more efficient utilization of available hardware resources by distributing the training loads across multiple GPUs. This feature is available through both the Python API and the command-line interface. To enable multi-GPU training specify the GPU device IDs you wish to use.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
# Train the model on 2 GPUs
results = model.train(data="coco8.yaml", epochs=100, img_size=640, device=[0, 1])
# Train the model on 2 most idle GPUs
results = model.train(data="coco8.yaml", epochs=100, img_size=640, device=[-1, -1])
```
```bash
vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640 device=0,1

vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640 device=-1,-1
```
### Idle GPU Training

Idle GPU Training enables automatic selection of the least utilized GPUs in multi-GPU systems, optimizing resource usage without manual GPU selection. This feature identifies GPUs based on utilization metrics and VRAM availablity.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
# Train using the single most idle GPU
results = model.train(data="coco8.yaml", epochs=100, img_size=640, device=-1)
# Train using the 2 most idle GPUs
results = model.train(data="coco8.yaml", epochs=100, img_size=640, device=[-1, -1])
```

```bash
vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640 device=-1

vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640 device=-1,-1
```

The auto-selection algorithm prioritized GPUs with:

1. Lower current utilization percentages
2. Higher available memory (free VRAM)
3. Lower temperature and power consumption

This feature is especially valuable in shared computing environments or when running multiple training jobs across different models. It automatically adapts to changing system conditions, ensuring optimal resource allocations without manual intervention.

### Apple Silicon MPS Training

With the support for Apple silicon chips integrated in Vayuvahana Technologies VajraV1 models, it is now possible to train your models on devices utilising the powerful Metal Performance Shaders (MPS) framework. The MPS offers a high-performance way of executing computation and image processing tasks on Apple's custom silicon.

To enable training on Apple silicon chips you need to specify 'mps' as your device when initiating training. Below is an example of how this can be done in Python and via CLI.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
results = model.train(data="coco8.yaml", epochs=100, img_size=640, device="mps")
```

```bash
vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640 device=mps
```

For more detailed guidance and advanced configuration options refer to the [Pytorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

### Resume Interrupted Trainings

```python
from vajra import Vajra

model = Vajra("path/to/last-vajra-v1-nano-det.pt")
results = model.train(resume=True)
```

```bash
vajra train resume model=path/to/last-vajra-v1-nano-det.pt
```

By setting `resume=True`, the `train` function will continue training from where it left off, using the state stored in the 'path/to/last-vajra-v1-nano-det.pt' file. If the `resume` argument is omitted or set to `False`, the trainer will start a new training session.

Checkpoints are saved at the end of every epoch by default, or at fixed intervals using the `save_period` argument, so you need to complete at least 1 epoch to resume a training run.

## Train Settings

The training settings for VajraV1 models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed and accuracy. Key training settings include batch size, learning rate, momentum and weight decay. Additionally, the choice of optimizer, loss function, and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

{% include "macros/train-args.md" %}

!!! info "Note on Batch-size Settings"
    The `batch` argument can be configured in three ways:

    - **Fixed Batch Size:** Set an integer value (e.g. `batch = 16`), specifying the number of images per batch directly.
    - **Auto Mode (60% GPU Memory):** Use `batch = -1` to automatically adjust batch size for approximately 60% CUDA memory utilization.
    - **Auto Mode with Utilization Fraction:** Set a fraction value (e.g. `batch = 0.7`) to adjust batch size based on the specified fraction of GPU memory usage.


## Augmentation Settings and Hyperparameters

Augmentation techniques are essential for improving the robustness and performance of the VajraV1 models by introducing variability into the training data, helping the model generalize better to unseen data. The following table outlines the purpose and effect of each augmentation argument:

{% include "macros/augmentation-args.md" %}

These settings can be adjusted to meet the specific requirements of the dataset and task at hand. Experimenting with different values can help find the optimal augmentation strategy that leads to the best model performance.

!!! info

    For more information about training augmentation operations, see the [reference section](../reference/data/augment.md).

## Logging

In training a VajraV1 model, you might find it valueable to keep track of the model's performance over time. This is where logging comes into play. Vayuvahana Technologies VajraV1 provides support for 3 types of loggers - [Comet](../integrations/comet.md), [ClearML](../integrations/clearml.md), and [TensorBoard](../integrations/tensorboad.md).

### Comet

[Comet](../integrations/comet.md) is a platform that allows data scientists and developers to track, compare, explain and optimize experiments and models. It provides functionalities such as real-time metrics, code diffs, and hyperparameter tracking.

To use Comet:

!!! example

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Remember to sign in to your Comet account on their website and get your API key. You will need to add this to your environment variables or your script to log your experiments.

### ClearML

[ClearML](https://clear.ml/) is an open source platform that automates tracking of experiments and helps with efficient sharing of resources. It is designed to help teams manage, execute, and reproduce their ML work more efficiently.

To use ClearML:

!!! example

    === "Python"

        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

After running this script, you will need to sign in to your ClearML account on the browser and authenticate your session.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization toolkit for Tensorflow. It allows you to visualize your Tensorflow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

To use TensorBoard in Google Colab:

!!! example

    === "CLI"

        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs # replace with 'runs' directory
        ```

To use TensorBoard locally run the below command and view results at `http://localhost:6006/`.

!!! example

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs # replace with 'runs' directory
        ```

This will load TensorBoard and direct it to the directory where your training logs are saved.

After setting up your logger, you can then proceed with your model trianing. All training metrics will be automatically logged in your chosen platform, and you can access these logs to monitor your model's performance over time, compare different models, and identify areas for improvement.

## FAQ

### How do I train an object detection model using Vayuvahana Technologies VajraV1?

To train an object detection model using Vayuvahana Technologies VajraV1, you can either use the Python API or the CLI. Below is an example for both:

!!! example "Single-GPU and CPU Training Example"

    === "Python"

        ```python
        from vajra import Vajra

        # Load a model
        model = Vajra("vajra-v1-nano-det.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, img_size=640)
        ```

    === "CLI"

        ```bash
        vajra detect train data=coco8.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640
        ```

For more details, refer to the [Train Settings](#train-settings) section.

### What are the key features of Vayuvahana Technologies VajraV1's Train mode?

The key features of Vayuvahana Technologies VajraV1 Train mode include:

- **Automatic Dataset Download:** Standard datasets like COCO, VisDrone, VOC, ImageNet are downloaded automatically on first use.
- **Multi-GPU Support:** Scale your training efforts seamlessly across multiple GPUs to expedite the process.
- **Hyperparameter Configuration:** The option to modify the hyperparameters through YAML configuration files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics and visualization of the learning process for better insights.

These features make training efficient and customizable to your needs. For more details see the [Key Features of Train Mode](#key-features-of-the-train-mode) section.

### What are the common training settings, and how do I configure them?

Vayuvahana Technologies VajraV1 allows you to configure a variety of training settings such as batch size, learning rate, epochs, and more through arguments. Here's a brief overview:

| Argument | Default | Description                                                            |
| -------- | ------- | ---------------------------------------------------------------------- |
| `model`  | `None`  | Keyword to build the model / Path to the model file for training.      |
| `data`   | `None`  | Path to the dataset configuration file (e.g., `coco8.yaml`).           |
| `epochs` | `100`   | Total number of training epochs.                                       |
| `batch`  | `16`    | Batch size, adjustable as integer or auto mode.                        |
| `img_size`  | `640`   | Target image size for training.                                     |
| `device` | `None`  | Computational device(s) for training like `cpu`, `0`, `0,1`, or `mps`. |
| `save`   | `True`  | Enables saving of training checkpoints and final model weights.        |

For an in-depth guide on training settings, check the [Train Settings](#train-settings) section.
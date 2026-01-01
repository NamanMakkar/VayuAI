# Vayuvahana Technologies VayuAI Hyperparameter Tuning Guide

## Introduction

Hyperparameter tuning is not just a one-time set-up but an iterative process aimed at optimizing the machine learning model's performance metrics, such as accuracy, precision, and recall. In the context of Vayuvahana Technologies VayuAI, these hyperparameters could range from learning rate to augmentations and architectural details such as the number of layers or types of activation functions used.

### What are Hyperparameters?

Hyperparameters are high-level, structural settings for the algorithm. They are set prior to the training phase and remain constant during it. Here are some commonly tuned hyperparameters:

- **Learning Rate** `lr0`: Determines the step size at each iteration while moving towards a minimum in the loss function.
- **Batch Size** `batch`: Number of images processed simultaneously in a forward pass.
- **Number of Epochs** `epochs`: An epoch is one complete forward and backward pass of all the training examples.
- **Architecture Specifics**: Such as channel counts, number of layers, types of activation functions etc.

### Genetic Evolution and Mutation Algorithms

Vayuvahana Technologies VayuAI uses genetic algorithms to optimize hyperparameters. Genetic algorithms are inspired by the mechanism of natural selection and genetics.

**Mutation**: In the context of Vayuvahana Technologies VayuAI, mutation helps in locally searching the hyperpameter space by applying small, random changes to existing hyperparameters, producing new candidates for evaluation.

## Preparing for Hyperparameter Tuning

Before you begin the tuning process, it's important to:

1. **Identify the Metrics**: Determine the metrics you will use to evaluate the model's performance. This could be AP50, F1-score, or others.
2. **Set the Tuning Budget**: Define how much computational resources you're willing to allocate. Hyperparameter tuning can be computationally intensive.

## Steps Involved

### Initialize Hyperparameters

Start with a reasonable set of initial hyperparameters. This could either be the default hyperparameters set by Vayuvahana Technologies VayuAI or something based on your domain knowledge or previous experiments.

### Mutate Hyperparameters

Use the `_mutate` method to produce a new set of hyperparameters based on the existing set. The Tuner class handles this process automatically.

### Train Model

Training is performed using the mutated set of hyperparameters. The training performance is the assessed using your chosen metrics.

### Evaluate Model

Use metrics like AP50, F1-score, or custom metrics to evaluate the model's performance. The evaluation process helps determine if the current hyperparameters are better than the previous ones.

### Log Results

It's crucial to log both the performance metrics and corresponding hyperparameters for future reference. Vayuvahana Technologies VayuAI automatically saves these results in CSV format.

### Repeat

The process is repeated until either the set number of iterations is reached or the performance metric is satisfactory. Each iteration builds upon the knowledge gained from teh previous runs.

## Default Search Space Description

The following table lists the default search space parameters for hyperparameter tuning in VajraV1. Each parameter has a specific value range defined by a tuple `(min, max)`.

| Parameter         | Type    | Value Range    | Description                                                                                                      |
| ----------------- | ------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `lr0`             | `float` | `(1e-5, 1e-1)` | Initial learning rate at the start of training. Lower values provide more stable training but slower convergence |
| `lrf`             | `float` | `(0.01, 1.0)`  | Final learning rate factor as a fraction of lr0. Controls how much the learning rate decreases during training   |
| `momentum`        | `float` | `(0.6, 0.98)`  | SGD momentum factor. Higher values help maintain consistent gradient direction and can speed up convergence      |
| `weight_decay`    | `float` | `(0.0, 0.001)` | L2 regularization factor to prevent overfitting. Larger values enforce stronger regularization                   |
| `warmup_epochs`   | `float` | `(0.0, 5.0)`   | Number of epochs for linear learning rate warmup. Helps prevent early training instability                       |
| `warmup_momentum` | `float` | `(0.0, 0.95)`  | Initial momentum during warmup phase. Gradually increases to the final momentum value                            |
| `box`             | `float` | `(0.02, 0.2)`  | Bounding box loss weight in the total loss function. Balances box regression vs classification                   |
| `cls`             | `float` | `(0.2, 4.0)`   | Classification loss weight in the total loss function. Higher values emphasize correct class prediction          |
| `hsv_h`           | `float` | `(0.0, 0.1)`   | Random hue augmentation range in HSV color space. Helps model generalize across color variations                 |
| `hsv_s`           | `float` | `(0.0, 0.9)`   | Random saturation augmentation range in HSV space. Simulates different lighting conditions                       |
| `hsv_v`           | `float` | `(0.0, 0.9)`   | Random value (brightness) augmentation range. Helps model handle different exposure levels                       |
| `degrees`         | `float` | `(0.0, 45.0)`  | Maximum rotation augmentation in degrees. Helps model become invariant to object orientation                     |
| `translate`       | `float` | `(0.0, 0.9)`   | Maximum translation augmentation as fraction of image size. Improves robustness to object position               |
| `scale`           | `float` | `(0.0, 0.9)`   | Random scaling augmentation range. Helps model detect objects at different sizes                                 |
| `shear`           | `float` | `(0.0, 10.0)`  | Maximum shear augmentation in degrees. Adds perspective-like distortions to training images                      |
| `perspective`     | `float` | `(0.0, 0.001)` | Random perspective augmentation range. Simulates different viewing angles                                        |
| `flipud`          | `float` | `(0.0, 1.0)`   | Probability of vertical image flip during training. Useful for overhead/aerial imagery                           |
| `fliplr`          | `float` | `(0.0, 1.0)`   | Probability of horizontal image flip. Helps model become invariant to object direction                           |
| `mosaic`          | `float` | `(0.0, 1.0)`   | Probability of using mosaic augmentation, which combines 4 images. Especially useful for small object detection  |
| `mixup`           | `float` | `(0.0, 1.0)`   | Probability of using mixup augmentation, which blends two images. Can improve model robustness                   |
| `copy_paste`      | `float` | `(0.0, 1.0)`   | Probability of using copy-paste augmentation. Helps improve instance segmentation performance 

## Custom Search Space Example

Here is how to define a search space and use the `model.tune()` method to utilize the `Tuner` class for hyperparameter tuning of VajraV1-nano-det on COCO8 for 30 epochs with an AdamW optimizer and skipping plotting, checkpointing and validation other than on the final epoch for faster Tuning.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")

        search_space = {
            "lr0": (1e-5, 1e-1),
            "degrees": (0.0, 45.0),
        }

        model.tune(
            data="coco8.yaml",
            epochs=30,
            iterations=300,
            optimizer="AdamW",
            space=search_space,
            plots=False,
            save=False,
            val=False,
        )
        ```

## Resuming an Interrupted Hyperparameter Tuning Session

You can resume an interrupted hyperparameter tuning session by passing `resume=True`. You can optionally pass the directory `name` used under `runs/{task}` ro resume. Otherwise it would resume the last interrupted session. You also need to provide all the previous training arguments including `data`, `epochs`, `iterations` and `space`.

!!! example "Using `resume=True` with `model.tune()`"

    ```python
    from vajra import Vajra
    model = Vajra("vajra-v1-nano-det.pt")

    search_space = {
        "lr0": (1e-5, 1e-1),
        "degrees": (0.0, 45.0),
    }

    results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, resume=True)

    results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, name="tune_exp", resume=True)
    ```

## Results

After you've successfully completed the hyperparameter tuning process, you will obtain several files and directories that encapsulate the results of the tuning. The following describes each:

### File Structure

Here's what the directory structure of the results will look like. Training directories like `train1/` contain individual tuning iterations, i.e. one model trained with one set of hyperparameters. The `tune/` directory contains tuning results from all the individual model trainings:

```plaintext
runs/
└── detect/
    ├── train1/
    ├── train2/
    ├── ...
    └── tune/
        ├── best_hyperparameters.yaml
        ├── best_fitness.png
        ├── tune_results.csv
        ├── tune_scatter_plots.png
        └── weights/
            ├── last-vajra-v1-nano-det.pt
            └── best-vajra-v1-nano-det.pt
```

### File Descriptions

#### best_hyperparameters.yaml

This YAML file contains the best-performing hyperparameters found during the tuning process. You can use this file to initialize future trainings with these optimized settings.

#### best_fitness.png

This is a plot displaying fitness (typically a performance metric like AP50) against the number of iterations. It helps you visualize how well the genetic algorithm performed over time.

#### tune_results.csv

A CSV file containing detailed results of each iteration during the tuning. Each row in the file represents one iteration, and it includes metrics like fitness score, precision, recall, as well as the hyperparameters used.

#### tune_scatter_plots.png

This file contains scatter plots generated from `tune_results.csv`, helping you visualize relationships between different hyperparameters and performance metrics. Note that hyperparameters initialized to 0 will not be tuned, such as `degrees` and `shear` below.

#### weights/

This directory contains the saved PyTorch models for the last and the best iterations during the hyperparameter tuning process.

- **`last-vajra-v1-nano-det.pt`**: These are the weights from the last epoch of training.
- **`best-vajra-v1-nano-det.pt`**: These are the weights for the iteration that achieved the best fitness score.

Using these results, you can make more informed decisions for your future model trainings and analyses. Feel free to consult these artifacts to understand how well your model performed and how you might improve it further.

## An Example of Hyperparameter Tuning on the COCO dataset

A detailed example for hyperparameter tuning on the COCO dataset can be found [here](https://github.com/NamanMakkar/VayuAI/blob/main/vajra/examples/hyperparameter_tuning.py)

This is an example where `model.tune()` has not been called and shows you another way to carry out hyperparameter tuning without using genetic algorithms.
 
```python
from vajra import Vajra
from vajra.configs import get_config, get_save_dir
import subprocess
import numpy as np
import torch
from pathlib import Path
from vajra.utils import LOGGER


copy_paste_hparams = np.arange(0., 0.41, 0.05)
mixup_hparams = np.arange(0., 0.3, 0.05)
fitness_list = {}
val_dfl_list = {}
i = 0
for cp in copy_paste_hparams:
    for mxp in mixup_hparams:
        LOGGER.info(f"\nITERATION {i}; copy_paste: {cp}; mixup: {mxp}\n")
        train_args = {
            "model": "vajra-v1-medium-det",
            "data": "coco.yaml",
            "project": "COCO_Manual_Tuning_SixEps",
            "name": "vajra-v1-medium-det-manual-tune",
            "batch": 120,
            "img_size": 640,
            "patience": 100,
            "epochs": 600,
            "stop_epoch": 7,
            "optimizer": "SGD",
            "lr0": 0.01,
            "device": 0,
            "seed": 0,
            "deterministic": True,
            "copy_paste": cp,
            "mixup": mxp,
        }
        save_dir = "COCO_Manual_Tuning_SixEps/vajra-v1-medium-det-manual-tune" if i==0 else f"COCO_Manual_Tuning_SixEps/vajra-v1-medium-det-manual-tune{i+1}"
        weights_dir = Path(f"{save_dir}/weights")
        cmd = ["vajra", "train", *(f"{k}={v}" for k, v in train_args.items())]
        return_code = subprocess.run(cmd, check=True).returncode
        checkpt_file = weights_dir / ("best-vajra-v1-medium-det.pt" if (weights_dir / "best-vajra-v1-medium-det.pt").exists() else "last-vajra-v1-medium-det.pt")
        metrics = torch.load(checkpt_file)["train_metrics"]
        val_dfl_loss = metrics["val/dfl_loss"]
        val_dfl_list[f"cp_{cp}_mxp_{mxp}"] = val_dfl_loss
        fitness = metrics["fitness"]
        fitness_list[f"cp_{cp}_mxp_{mxp}"] = fitness
        i += 1
        LOGGER.info(f"\nFitness: {fitness}\n")
        LOGGER.info(f"\nVal DFL Loss: {val_dfl_loss}\n")
        LOGGER.info(f"\nITERATION {i - 1}: copy_paste: {cp}; mixup: {mxp}\n")

max_fitness = max(fitness_list, key=fitness_list.get)
min_val_dfl = min(val_dfl_list, key=val_dfl_list.get)
LOGGER.info(f"\nKey for min dfl_loss: {min_val_dfl}; Val DFL loss: {val_dfl_list[min_val_dfl]}; Fitness: {fitness_list[min_val_dfl]}\n")
LOGGER.info(f"\nKey for max fitness: {max_fitness}; Val DFL loss: {val_dfl_list[max_fitness]}; Fitness: {fitness_list[max_fitness]}\n") 
for (k, v) in val_dfl_list.items():
    LOGGER.info(f"\n{k}: val dfl loss: {v}; Val Accuracy: {fitness_list[k]}\n")
```

## Conclusion

The hyperparameter tuning process in Vayuvahana Technologies VayuAI is simplified yet powerful, thanks to its genetic algorithm-based approach focused on mutation. Following the steps outlined in this guide will assist you in systematically tuning your model to achieve better performance.

## FAQ

### How do I optimize the learning rate for VajraV1 during hyperparameter tuning?

To optimize the learning rate for the VajraV1, start by setting an initial learning rate using the lr0` parameter. Common values range from `0.001` to `0.01`. During the hyperparameter tuning process, this value will be mutated to find the optimal setting. You can utilize the `model.tune()` method to automate this process. For example:

!!! example

    === "Python"

        ```python
        from vajra import Vajra
        model = Vajra("vajra-v1-nano-det.pt")

        model.tune(data="coco8.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

### What are the benefits of using genetic algorithms for hyperparameter tuning in VayuAI?

Genetic algorithms in Vayuvahana Technologies VayuAI provide a robust method for exploring the hyperparameter space, leading to highly optimized model performance. Key benefits include:

- **Efficient Search**: Genetic algorithms like mutation can quickly explore a large set of hyperparameters.
- **Avoiding Local Minima**: By introducing randomness, they help in avoiding local minima, ensuring better global optimization.
- **Performance Metrics**: They adapt based on performance metrics such as AP50 and F1-score.

### How long does the hyperparameter tuning process take?

The time required for hyperparameter tuning with Vayuvahana Technologies VayuAI largely depends on several factors such as the size of the dataset, the complexity of the model architecture, the number of iterations, the number of epochs and the computational resources available. Tuning VajraV1 on a dataset like the COCO dataset can take several hours to days, depending on your hardware.


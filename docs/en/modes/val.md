# Model Validation with Vayuvahana Technologies VayuAI

## Introduction 

Validation is a critical step in the machine learning pipeline, allowing you to assess the quality of your trained models. Val mode i Vayuvahana Technologies VayuAI provides a robust suite of tools and metrics for evaluating the performance of your object detection models. This guide serves as a complete resource for understanding how to effectively use the Val mode to ensure that your models are both accurate and reliable.

## Why Validate with Vayuvahana Technologies VayuAI?

Here's why using VayuAI's Val mode is advantageous:

- **Precision:** Get accurate metrics like mAP50, mAP75, and mAP50-95 to comprehensively evaluate your model.
- **Convenience:** Utilize built-in features that remember training settings, simplifying the validation process.
- **Flexibility:** Validate your model with the same or different datasets and image sizes.
- **Hyperparameter Tuning:** Use validation metrics to fine-tune your model for better performance.

### Key Features of Val Mode

These are the notable functionalities offered by VayuAI's Val mode:

- **Automated Settings:** Models remember their training configurations for straightforward validation.
- **Multi-Metric Support:** Evaluate your model based on a range of accuracy metrics.
- **CLI and Python API:** Choose from command-line interface or Python API based on your preference for validation.
- **Data Compatibility:** Works seamlessly with datasets used during the training phase as well as custom datasets.

!!! tip

    * VajraV1 models automatically remember their training settings, so you can validate a model at the same image size and on the original dataset easily with just `vajra val model=vajra-v1-nano-det.pt` or `model("vajra-v1-nano-det.pt").val()`

## Usage Examples

Validate trained VajraV1 model accuracy on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes. See Arguments section below for a full list of validation arguments.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Load a model
        model = Vajra("vajra-v1-nano-det.pt")  # load an official model
        model = Vajra("path/to/best-vajra-v1-nano-det.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        ```

    === "CLI"

        ```bash
        vajra detect val model=vajra-v1-nano-det.pt      # val official model
        vajra detect val model=path/to/best-vajra-v1-nano-det.pt # val custom model
        ```

## Arguments for VajraV1 Model Validation

When validating VajraV1 models, several arguments can be fine-tuned to optimize the evaluation process. These arguments control aspects such as input image size, batch processing, and performance thresholds. Below is a detailed breakdown of each argument to help you customize your validation settings effectively.

{% include "macros/validation-args.md" %}

Each of these settings plays a vital role in the validation process, allowing for a customizable and efficient evaluation of VajraV1 models. Adjusting these parameters according to your specific needs and resources can help achieve the best balance between accuracy and performance.

### Example Validation with Arguments

The below examples showcase model validation with custom arguments in Python and CLI.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Load a model
        model = Vajra("vajra-v1-nano-det.pt")

        # Customize validation settings
        metrics = model.val(data="coco8.yaml", img_size=640, batch=16, conf=0.25, iou=0.6, device=0)
        ```

    === "CLI"

        ```bash
        vajra val model=vajra-v1-nano-det.pt data=coco8.yaml img_size=640 batch=16 conf=0.25 iou=0.6 device=0
        ```

## FAQ

### How do I validate my VajraV1 model with VayuAI?

To validate your VajraV1 model, you can use the Val mode. For example, using the Python API, you can load a model and run validation with:

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-det.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # map50-95
```

Alternatively, you can use the command-line interface (CLI):

```bash
vajra val model=vajra-v1-nano-det.pt
```

For further customization you can adjust various arguments like `img_size`, `batch`, and `conf` in both Python and CLI modes. Check the [Arguments for Vajra Model Validation](#arguments-for-vajrav1-model-validation) section for full list of parameters.

### What metrics can I get from VajraV1 model validation?

VajraV1 model validation provides several key metrics to assess model performance. These include:

- mAP50 (mean Average Precision at IoU threshold 0.5)
- mAP75 (mean Average Precision at IoU threshold 0.75)
- mAP50-95 (mean Average Precision across multiple IoU thresholds from 0.5 to 0.95)

Using the Python API, you can access these metrics as follows:

```python
metrics = model.val()  # assumes `model` has been loaded
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # list of mAP50-95 for each category
```

For a complete performance evaluation, it is crucial to review all these metrics. For more details refer to the [Key Features of Val Mode](#key-features-of-val-mode).

### What are the advantages of using Vayuvahana Technologies VayuAI for validation?

Using Vayuvahana Technologies VayuAI for validation provides several advantages:

- **Precision:** VayuAI offers accurate performance metrics including mAP50, mAP75, and mAP50-95.
- **Convenience:** The models remember their training settings, making validation straightforward.
- **Flexibility:** You can validate against the same or different datasets and image sizes.
- **Hyperparameter Tuning:** Validation metrics help in fine-tuning models for better performance.

These benefits ensure that your models are evaluated thoroughly and can be optimized for superior results. Learn more about these advantages in the [Why Validate with Vayuvahana Technologies VayuAI](#why-validate-with-vayuvahana-technologies-vayuai) section.

### Can I validate my VajraV1 model using a custom dataset?

Yes, you can validate your VajraV1 model using a custom dataset. Specify the `data` argument with the path to your dataset configuration file. This file should include paths to the validation data, class names, and other relevant details.

Example in Python:

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-det.pt")

# Validate with a custom dataset
metrics = model.val(data="path/to/your/custom_dataset.yaml")
print(metrics.box.map)  # map50-95
```

Example using CLI:

```bash
vajra val model=vajra-v1-nano-det.pt data=path/to/your/custom_dataset.yaml
```

For more customizable options during validation, see the [Example Validation with Arguments](#example-validation-with-arguments) section.

### How do I save validation results to a JSON file in VayuAI?

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-det.pt")

# Save validation results to JSON
metrics = model.val(save_json=True)
```

Example using CLI:

```bash
vajra val model=vajra-v1-nano-det.pt save_json=True
```

This functionality is particularly useful for further analysis or integration with other tools. Check the [Arguments for VajraV1 Model Validation](#arguments-for-vajrav1-model-validation) for more details.
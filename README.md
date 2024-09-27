[Vayuvahana Technologies Private Limited VajraV1] (https://github.com/NamanMakkar/VayuAI) is a 
state-of-the-art (SOTA) real time object detection model inspired by the YOLO model architectures. VajraV1 is a family of fast, lightweight models that can be used for a variety of
tasks like object detection and tracking, instance segmentation, oriented object detection, pose detection, and image classification.

## <div align="center">Enterprise License</div>
To request for an Enterprise License please get in touch [Email](mailto:namansingh2803@gmail.com)

## <div align="center">Performance</div>
Details to be published

## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

Git clone the VayuAI package including all [requirements](https://github.com/NamanMakkar/VayuAI/pyproject.toml) in a [**Python>=3.8**](https://www.python.org) environment.

```bash
git clone https://github.com/NamanMakkar/VayuAI.git
pip install .
```
</details>

<details open>
<summary>Usage</summary>

### CLI
Vajra can be used in the Command Line Interface with a `vajra` or `vayuvahan` or `vayuai`
command:

```bash
vajra predict model=vajra-v1-nano-det img_size=640 source="path/to/source.jpg"
```

### Python
Vajra can also be used directly in a Python environment, and accepts the same arguments as in the CLI example above:

```python
from vajra import Vajra, VajraDEYO
model = Vajra("vajra-v1-nano-det")

train_results = model.train(
    data="coco8.yaml",
    epochs=100,
    img_size=640,
    device="cpu"
)

metrics = model.val()
results = model("path/to/img.jpg")
results[0].show()

path = model.export(format="onnx")
```
</details>

## Model Architectures

- VajraV1-det
- VajraV1-cls
- VajraV1-pose
- VajraV1-seg
- VajraV1-obb
- VajraV1-DEYO-det
- SAM
- EfficientNetV1
- EfficientNetV2
- VajraEffNetV1
- VajraEffNetV2
- ConvNeXtV1
- ConvNeXtV2
- ResNet
- ResNeSt
- EdgeNeXt
- ME-NeSt
- MixConvNeXt

## Tasks Supported

- detect
- small_obj_detect
- classify
- pose
- obb
- segment
- multilabel_classify
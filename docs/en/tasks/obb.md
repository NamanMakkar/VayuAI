# Oriented Bounding Boxes Object Detection

Oriented object detection goes a step further than standard object detection by introducing an extra parameter (an angle) to locate the object more accurately in an image by calculating the orientation of the bounding box.

The output of an oriented object detector is a set of rotated bounding boxes that precisely enclose the objects in the image, along with class labels and confidence scores for each box. Oriented bounding boxes are particularly useful when objects appear at various angles, such as in aerial imagery, where traditional axis-aligned boxes may include unnecessary background.

## Model

The VajraV1 pretrained OBB models are shown here which are pretrained on the [DOTAv1](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/DOTAv1.yaml) dataset.

## Train

Train VajraV1n-obb on the DOTA8 dataset for 100 epochs. 

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-obb") #Build a new model
model = Vajra("vajra-v1-nano-obb.pt") #Load a pretrained model
model = Vajra("vajra-v1-nano-obb").load("vajra-v1-nano-obb.pt") #Build a new model and load pretrained weights

result = model.train(data="dota8.yaml", epochs=100, img_size=640)
```

You can also train the model using CLI:

```bash
# Build a new model and train
vajra obb train model=vajra-v1-nano-obb data=dota8.yaml img_size=640 epochs=100

# Load a pretrained model and train
vajra obb train model=vajra-v1-nano-obb.pt data=dota8.yaml img_size=640 epochs=100
```
### Dataset format

OBB dataset format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1, following this structure:

```
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Internally losses and outputs are processed in the `xywhr` format, which represents the bounding box's center point (xy), width, height and rotation.

## Val

Validate a trained VajraV1 obb model on the DOTA8 dataset. No arguments are needed as the model retains its training data and arguments as model attributes.

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-obb.pt") # COCO pretrained model
model = Vajra("path/to/best-vajra-v1-nano-obb.pt") # Custom model

# Validate the model
metrics = model.val(data="dota8.yaml") # dataset and settings remembered from training
metrics.box.map # map50-95
metrics.box.map50 # map50
metrics.box.map75 # map75
metrics.box.maps # a list containing map50-95 of each category
```

```bash
vajra obb val model=vajra-v1-nano-obb.pt data=dota8.yaml
vajra obb val model=path/to/best-vajra-v1-nano-obb.pt data=dota8.yaml
```

## Predict

Use a trained VajraV1n-obb model to run predictions on images.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-obb.pt") # Load a DOTAv1 pretrained model
model = Vajra("path/to/best/vajra-v1-nano-obb.pt")

results = model("path/to/img.jpg")

for result in results:
    xywhr = result.obb.xywhr
    xyxyxyxy = result.obb.xyxyxyxy
    names = [result.name[cls.item()] for cl in result.obb.cls.int()]
    confs = result.obb.conf
```

```bash
vajra obb predict model=vajra-v1-nano-obb.pt source="path/to/img.jpg"

vajra obb predict model=path/to/best-vajra-v1-nano-obb.pt source="path/to/img.jpg"
```

## Export

Export a VajraV1n-obb model to a different format like ONNX, TensorRT, CoreML etc.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-obb.pt")
model = Vajra("path/to/best-vajra-v1-nano-obb.pt")

model.export(format="onnx")
```

```bash
vajra export model=vajra-v1-nano-obb.pt
```
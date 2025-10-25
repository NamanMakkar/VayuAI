# Object Detection

Object Detection is a task that involves locating objects and classifying them in an image or video stream.

An object detector outputs a set of bounding boxes that enclose the objects in the image along with the class labels and confidence scores for each box. Object Detection is required for identifying objects of interest in a scene.

!!! tip
    VajraV1 Detection models use the `-det` suffix, i.e `vajra-v1-nano-det.pt` and are pretrained on the [COCO](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/coco.yaml) dataset. These are the default VajraV1 models.

## Models

The VajraV1 detection models are shown here. VajraV1 detection, segmentation and pose models have been pretrained on the [COCO](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/coco.yaml) dataset. The Classification models is being trained on the [ImageNet](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/) dataset.

## Train

Train the VajraV1 model on the COCO8 dataset for 100 epochs in python:

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-det") #Build a new model
model = Vajra("vajra-v1-nano-det.pt") #Load a pretrained model
model = Vajra("vajra-v1-nano-det").load("vajra-v1-nano-det.pt") #Build a new model and load pretrained weights

result = model.train(data="coco8.yaml", epochs=100, img_size=640)
```
You can also train the model using CLI:

```bash
# Build a new model and train
vajra detect train model=vajra-v1-nano-det data=coco8.yaml img_size=640 epochs=100

# Load a pretrained model and train
vajra detect train model=vajra-v1-nano-det.pt data=coco8.yaml img_size=640 epochs=100
```

## Val

Validate trained VajraV1 detection model on the dataset. No arguments needed as the model retains its training data and arguments as model attributes.

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-det.pt") # COCO pretrained model
model = Vajra("path/to/best-vajra-v1-nano-det.pt") # Custom model

# Validate the model
metrics = model.val() # dataset and settings remembered from training
metrics.box.map # map50-95
metrics.box.map50 # map50
metrics.box.map75 # map75
metrics.box.maps # a list containing map50-95 of each category
```

```bash
vajra detect val model=vajra-v1-nano-det.pt 
vajra detect val model=path/to/best-vajra-v1-nano-det.pt
```

## Predict
Use a trained VajraV1 model to run predictions on images

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt") # Load a COCO pretrained model
model = Vajra("path/to/best/vajra-v1-nano-det.pt")

results = model("path/to/img.jpg")

for result in results:
    xywh = result.boxes.xywh
    xywhn = result.boxes.xywhn
    xyxy = result.boxes.xyxy
    xyxyn = result.boxes.xyxyn
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    confs = result.boxes.conf
```

```bash
vajra detect predict model=vajra-v1-nano-det.pt source="path/to/img.jpg"

vajra detect predict model=path/to/best-vajra-v1-nano-det.pt source="path/to/img.jpg"
```

## Export

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model = Vajra("path/to/best-vajra-v1-nano-det.pt")

model.export(format="onnx")
```

```bash
vajra export model=vajra-v1-nano-det.pt
```





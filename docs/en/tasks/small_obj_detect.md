# Small Object Detection

Small Object Detection takes object detection a step further by utilising keypoint detection for the purpose of detecting the centroid of each bounding box in addition to the bounding boxes. This is achieved by utilising Pose models for training and Detection models for validation. The Pose models have a pose branch for keypoint detection and a detection branch for bounding box regression and classification. The weights from the pose model are transferred to the detection model during validation. This technique enhances object detection for small objects.

## Train

Train a VajraV1 detection model using small_obj_detect on the VisDrone dataset

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-det") #Build a new model
model = Vajra("vajra-v1-nano-det.pt") #Load a pretrained model
model = Vajra("vajra-v1-nano-det").load("vajra-v1-nano-det.pt") #Build a new model and load pretrained weights

result = model.train(task = "small_obj_detect", data="VisDrone.yaml", epochs=100, img_size=640)
```

## Val

Validate the detection model on the dataset.

```python
from vajra import Vajra

# Load a model
model = Vajra("path/to/best-vajra-v1-nano-det.pt") # Custom model

# Validate the model
metrics = model.val() # dataset and settings remembered from training
metrics.box.map # map50-95
metrics.box.map50 # map50
metrics.box.map75 # map75
metrics.box.maps # a list containing map50-95 of each category
```

## Predict

Use the trained model to run predictions on images

```python
from vajra import Vajra

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
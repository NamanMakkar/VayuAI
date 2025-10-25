# Pose Estimation

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. Keypoints can represent various parts of the object such as joints, landmarks or other distinctive features. The locations of the keypoints are usually represented as a set of 2D '[x, y]' or 3D '[x, y, visible]' coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

The VajraV1 Pose models use the '-pose' suffix, i.e. 'vajra-v1-nano-pose.pt'. These models are trained on the [COCO keypoints](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

In the default VajraV1 pose model, there are 17 keypoints, each representing a different part of the human body. Here is a mapping of each index to its respective body joint:

0. Nose
1. Left Eye
2. Right Eye
3. Left Ear
4. Right Ear
5. Left Shoulder
6. Right Shoulder
7. Left Elbow
8. Right Elbow
9. Left Wrist
10. Right Wrist
11. Left Hip
12. Right Hip
13. Left Knee
14. Right Knee
15. Left Ankle
16. Right Ankle

## Models

The VajraV1 Pose Estimation models are shown here. VajraV1 detection, segmentation and pose models have been pretrained on the [COCO](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/coco.yaml) dataset. The Classification models is being trained on the [ImageNet](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/) dataset.

## Train

Train a VajraV1-pose model on the COCO8-pose dataset.

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-pose") #Build a new model
model = Vajra("vajra-v1-nano-pose.pt") #Load a pretrained model
model = Vajra("vajra-v1-nano-pose").load("vajra-v1-nano-pose.pt") #Build a new model and load pretrained weights

result = model.train(data="coco8-pose.yaml", epochs=100, img_size=640)
```

You can also train the model using CLI:

```bash
# Build a new model and train
vajra detect train model=vajra-v1-nano-pose data=coco8-pose.yaml img_size=640 epochs=100

# Load a pretrained model and train
vajra detect train model=vajra-v1-nano-pose.pt data=coco8-pose.yaml img_size=640 epochs=100
```

## Val

Validate trained VajraV1-pose model on the COCO8-pose dataset. No arguments needed as the model retains its training data and arguments as model attributes.

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-pose.pt") # COCO pretrained model
model = Vajra("path/to/best-vajra-v1-nano-pose.pt") # Custom model

# Validate the model
metrics = model.val() # dataset and settings remembered from training
metrics.box.map # map50-95(B)
metrics.box.map50 # map50(B)
metrics.box.map75 # map75(B)
metrics.box.maps # a list containing map50-95(B) of each category

metrics.pose.map  # map50-95(P)
metrics.pose.map50  # map50(P)
metrics.pose.map75  # map75(P)
metrics.pse.maps  # a list contains map50-95(P) of each category
```

```bash
vajra pose val model=vajra-v1-nano-pose.pt 
vajra pose val model=path/to/best-vajra-v1-nano-pose.pt
```

## Predict

Use a trained VajraV1 Pose model to run predictions on images

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-pose.pt") # Load a COCO pretrained model
model = Vajra("path/to/best/vajra-v1-nano-pose.pt")

results = model("path/to/img.jpg")

for result in results:
    xy = result.keypoints.xy
    xyn = result.keypoints.xyn
    masks = result.keypoints.data
```

```bash
vajra pose predict model=vajra-v1-nano-pose.pt source="path/to/img.jpg"

vajra pose predict model=path/to/best-vajra-v1-nano-pose.pt source="path/to/img.jpg"
```

## Export

Export a VajraV1 Pose model to a different format like ONNX, TensorRT, CoreML etc. This allows you to deploy your model on various platforms and devices for real time inference.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-pose.pt")
model = Vajra("path/to/best-vajra-v1-nano-pose.pt")

model.export(format="onnx")
```

```bash
vajra export model=vajra-v1-nano-pose.pt
```
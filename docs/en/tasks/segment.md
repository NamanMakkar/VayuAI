# Instance Segmentation

Instance Segmentation involves identifying individual objects in an image and segmenting them from the rest of the image.

An instance segmentation model outputs a set of masks or contours that outline each object in the image, along with class labels and confidence scores for each object. Instance segmentation is important for knowing where the objects are in an image as well as their exact shape.

## Models

The VajraV1 pretrained segmentation models are shown here. The VajraV1 detection, segmentation and pose models have been trained on the [COCO](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/coco.yaml) dataset. While the VajraV1 classification models are being trained on the [ImageNet](https://github.com/NamanMakkar/VayuAI/vajra/configs/datasets/) dataset.

## Train

Train the VajraV1 segmentation model on the COCO8-seg dataset for 100 epochs:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-seg")
model = Vajra("vajra-v1-nano-seg.pt")
model = Vajra("vajra-v1-nano-seg").load("vajra-v1-nano-seg.pt")

result = model.train(data="coco8-seg.yaml", img_size=640, epochs=100)
```

You can also train using CLI:

```bash
vajra segment train model=vajra-v1-nano-seg data=coco8-seg.yaml epochs=100 img_size=640

vajra segment train model=vajra-v1-nano-seg.pt data=coco8-seg.yaml epochs=100 img_size=640
```

## Val

```python
from vajra import Vajra

# Load a model
model = Vajra("vajra-v1-nano-seg.pt") # COCO pretrained model
model = Vajra("path/to/best-vajra-v1-nano-seg.pt") # Custom model

# Validate the model
metrics = model.val() # dataset and settings remembered from training
metrics.box.map # map50-95(B)
metrics.box.map50 # map50(B)
metrics.box.map75 # map75(B)
metrics.box.maps # a list containing map50-95(B) of each category
metrics.seg.map # map50-95(M)
metrics.seg.map50 # map50(M)
metrics.seg.map75 # map75(M)
metrics.seg.maps # a list containing map50-95(M) of each category
```

```bash
vajra segment val model=vajra-v1-nano-seg.pt 
vajra segment val model=path/to/best-vajra-v1-nano-seg.pt
```

## Predict
Use a trained VajraV1n-seg model to run predictions on images.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-seg.pt") # Load a COCO pretrained model
model = Vajra("path/to/best/vajra-v1-nano-seg.pt")

results = model("path/to/img.jpg")

for result in results:
    xy = result.masks.xy
    xyn = result.masks.xyn
    masks = result.masks.data
```

```bash
vajra segment predict model=vajra-v1-nano-seg.pt source="path/to/img.jpg"

vajra segment predict model=path/to/best-vajra-v1-nano-seg.pt source="path/to/img.jpg"
```

## Export

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-seg.pt")
model = Vajra("path/to/best-vajra-v1-nano-seg.pt")

model.export(format="onnx")
```

```bash
vajra export model=vajra-v1-nano-seg.pt
```
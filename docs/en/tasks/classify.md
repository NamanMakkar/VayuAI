# Image Classification

Image Classification involves classifying an image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where the objects of that class are located, what their exact shape is or whether it belongs to multiple classes or not.

## Train

Train a VajraV1 model on the Imagenet Dataset:

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-cls")

model.train(data="../data/imagenet", epochs=200, batch=256, img_size=224)
```

```bash
vajra classify model=vajra-v1-nano-cls data=imagenet.yaml epochs=200 batch=256 img_size=244
```

## Val

Validate the trained VajraV1 classification model on the ImageNet dataset. 

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-cls.pt")
model = Vajra("path/to/best-vajra-v1-nano-cls.pt")

metrics=model.val()
metrics.top5
metrics.top1
```

```bash
vajra classify val model=vajra-v1-nano-cls.pt
vajra classify val model=path/to/best-vajra-v1-nano-cls.pt
```

## Predict

Use a pretrained VajraV1 model to run predictions on images

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-cls.pt")
model = Vajra("path/to/best/vajra-v1-nano-cls.pt")
results = model("path/to/img.jpg")
```

## Export

Export VajraV1-cls models to the ONNX format using the code below:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-cls.pt")
model = Vajra("path/to/best-vajra-v1-nano-cls.pt")

model.export(format="onnx")
```

```bash
vajra classify mode=export model=vajra-v1-nano-cls.pt 
```

<div align="center">
<img src="./vajra/assets/Vayuvahana_logo.png" alt="Vayuvahana Technologies Private Limited Logo" width="450">
</div>

Vayuvahana Technologies Private Limited [VajraV1](https://github.com/NamanMakkar/VayuAI) is a 
state-of-the-art (SOTA) real time object detection model inspired by the YOLO model architectures. VajraV1 is a family of fast, lightweight models that can be used for a variety of
tasks like object detection and tracking, instance segmentation, oriented object detection, pose detection, and image classification.

## <div align="center">Enterprise License</div>
To request for an Enterprise License please get in touch via [Email](mailto:namansingh2803@gmail.com)

## <div align="center">Performance on COCO Dataset</div>

| Model                                                                                | Size (pixels) | mAP<sup>val<br>50-95</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|----------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.2/vajra-v1-nano-det.pt) | 640           | 44.3                       | 1.0                                                 | 3.78       | 13.7       |
| [VajraV1-small-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.2/vajra-v1-small-det.pt)  | 640           | 50.4                       | 1.1                                                 | 11.58      | 47.9      |
| [VajraV1-medium-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.2/vajra-v1-medium-det.pt)  | 640           | 52.7                           | 1.5                                                 | 20.29      | 94.5      |
| [VajraV1-large-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.2/vajra-v1-medium-det.pt)  | 640           | 53.7                           | 1.8                                                 | 24.63      | 115.2      |
| [VajraV1-xlarge-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.2/vajra-v1-xlarge-det.pt)  | 640           | 56.2                           | 3.2                                                 | 72.7      | 208.3     |


<p align="center">
<img width="90%" src="./vajra/assets/vajra_v1_coco_dataset_performance.png" alt="VajraV1 Performance on COCO Dataset">
</p>

## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

Git clone the VayuAI SDK including all [requirements](https://github.com/NamanMakkar/VayuAI/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org) environment.

```bash
git clone https://github.com/NamanMakkar/VayuAI.git
cd VayuAI
pip install .
```
</details>

<details open>
<summary>Usage</summary>

### CLI
Vajra can be used in the Command Line Interface with a `vajra` or `vayuvahana` or `vayuai`
command:

```bash
vajra predict model=vajra-v1-nano-det img_size=640 source="path/to/source.jpg"
```

### Python
Vajra can also be used directly in a Python environment, and accepts the same arguments as in the CLI example above:

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-det")

train_results = model.train(
    data="coco8.yaml",
    epochs=100,
    img_size=640,
    device="cpu",
    weight_decay=0.,
)

metrics = model.val()
results = model("path/to/img.jpg")
results[0].show()

path = model.export(format="onnx")
```

Pretrained COCO weights can also be used for model inference 

```python
from vajra import Vajra
model = Vajra("vajra-v1-xlarge-det.pt")
results = model("path/to/img.jpg")
results[0].show()

path = model.export(format="engine", device=0, half=True)
```
</details>

## Model Architectures

✅ VajraV1-det  
✅ VajraV1-cls  
✅ VajraV1-pose  
✅ VajraV1-seg  
✅ VajraV1-obb
✅ SAM  
✅ SAM2  
✅ FastSAM  
✅ MobileSAM 
✅ EfficientNetV1  
✅ EfficientNetV2  
✅ VajraEffNetV1  
✅ VajraEffNetV2  
✅ ConvNeXtV1  
✅ ConvNeXtV2  
✅ ResNet  
✅ ResNeSt  
❌ ResNeXt (Coming Soon!)  
❌ ResNetV2 (Coming Soon!)  
✅ EdgeNeXt  
✅ ME-NeSt  
✅ VajraME-NeSt  
✅ MixConvNeXt  
❌ ViT (Coming Soon!)  
❌ Swin (Coming Soon!)  
❌ SwinV2 (Coming Soon!)  

## Tasks Supported

✅ detect  
✅ small_obj_detect  
✅ classify  
✅ multilabel_classify  
✅ pose  
✅ obb  
✅ segment  
✅ world  

## Model Architecture Details

To be published

## Acknowledgements

- [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [https://github.com/ouyanghaodong/DEYOv1.5](https://github.com/ouyanghaodong/DEYOv1.5)
- [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
- [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

## <div align="center">License</div>

Vayuvahana Technologies Private Limited offers two licensing options:

- **AGPL-3.0 License**: This is an [OSI-approved](https://opensource.org/license) open-source
license for researchers for the purpose of promoting collaboration. See the [LICENSE](https://github.com/NamanMakkar/VayuAI/blob/main/LICENSE) file for details.

- **Enterprise License**: This license is designed for commercial use and enables integration of 
VayuAI software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your product requires embedding the software for commercial purposes or require access to more capable enterprise AI models in the future, reach out via [Email](mailto:namansingh2803@gmail.com).
# COCO Dataset

The [COCO](https://cocodataset.org/#home) (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and pose estimation tasks.

The [COCO-Seg](https://cocodataset.org/#home) dataset is an extension of the COCO dataset, it is specially designed to aid research in object instance segmentation. It uses the same images as the COCO dataset but introduces more detailed segmentation annotations. The dataset is a crucial resource for researchers and developers working on instance segmentation tasks, especially for training VajraV1 models.

## COCO Pretrained Models Perf

### COCO Detection Perf

| Model                                                                                | Size (pixels) | mAP<sup>val<br>50-95</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|----------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-nano-det.pt) | 640           | 44.3                       | 1.0                                                 | 3.78       | 13.7       |
| [VajraV1-small-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-small-det.pt)  | 640           | 50.4                       | 1.1                                                 | 11.58      | 47.9      |
| [VajraV1-medium-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-medium-det.pt)  | 640           | 52.7                           | 1.5                                                 | 20.29      | 94.5      |
| [VajraV1-large-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-large-det.pt)  | 640           | 53.7                           | 1.8                                                 | 24.63      | 115.2      |
| [VajraV1-xlarge-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-xlarge-det.pt)  | 640           | 56.2                           | 3.2                                                 | 72.7      | 208.3     |

### COCO Segmentation Perf

| Model                                                                                | Size (pixels) | Box mAP<sup>val<br>50-95</sup> | Mask mAP<sup>val<br>50-95</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|-------------------------------|-------------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-nano-seg.pt) | 640           | 43.6                          | 35.8                             | 1.2                                                 | 4.03       | 17.6      |
| [VajraV1-small-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-small-seg.pt) | 640           | 50.2                          | 40.5                             | 1.2                                                 | 12.23      | 61.9      |
| [VajraV1-medium-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-medium-seg.pt) | 640           | 52.6                          | 42.3                             | 1.7                                                 | 22.6      | 149.9      |
| [VajraV1-large-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-large-seg.pt) | 640           | 53.6                          | 43.1                             | 2.0                                                 | 26.93      | 170.6     |
| [VajraV1-xlarge-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-xlarge-seg.pt) | 640           | 55.7                          | 44.5                             | 3.4                                                 | 75       | 278.1     |

## Key Features

### COCO Detection Dataset

- COCO contains 330K images, with 200K images having annotations for object detection, segmentation, and captioning tasks.
- The dataset comprises 80 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as umbrellas, handbags, and sports equipment.
- Annotations include object bounding boxes, segmentation masks, and captions for each image.
- COCO provides standardized evaluation metrics like mean Average Precision (mAP) for object detection, and mean Average Recall (mAR) for segmentation tasks, making it suitable for comparing model performance.

### COCO-Seg Dataset

- COCO-Seg retains the original 330K images from COCO.
- The dataset consists of the same 80 object categories found in the original COCO dataset.
- Annotations now include more detailed instance segmentation masks for each object in the images.
- COCO-Seg provides standardized evaluation metrics like mean Average Precision (mAP) for object detection, and mean Average Recall (mAR) for instance segmentation tasks, enabling effective comparison of model performance.

## Dataset Structure

The COCO dataset is split into three subsets:

1. **Train2017**: This subset contains 118K images for training object detection, segmentation, and captioning models.
2. **Val2017**: This subset has 5K images used for validation purposes during model training.
3. **Test2017**: This subset consists of 20K images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7384) for performance evaluation.

## Applications

### COCO Detection Dataset

The COCO dataset is widely used for training and evaluating deep learning models in object detection (such as [Vayuvahana Technologies VajraV1](../models/vajrav1.md), Ultralytics YOLO, [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [SSD](https://arxiv.org/abs/1512.02325)), [instance segmentation] (such as [Mask R-CNN](https://arxiv.org/abs/1703.06870)), and keypoint detection (such as [OpenPose](https://arxiv.org/abs/1812.08008)). The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

### COCO-Seg Dataset

COCO-Seg is widely used for training and evaluating deep learning models in instance segmentation, such as VajraV1 models and YOLO models. The large number of annotated images, the diversity of object categories, and the standardized evaluation metrics make it an indispensable resource for computer vision researchers and practitioners.

## Usage

### COCO Detection Dataset

You can train VajraV1 models on the COCO dataset from scratch by following the steps given in the code snippet below:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det")

model.train(data="coco.yaml", device=[0, 1, 2, 3, 4, 5, 6, 7], img_size=640, epochs=600, batch=128)
```

```bash
vajra train model=vajra-v1-nano-det data=coco.yaml device=[0, 1, 2, 3, 4, 5, 6, 7] img_size=640 epochs=600 batch=128
```

### COCO-Seg Dataset

Similarly for segmentation, VajraV1-Seg models can be trained from scratch on the COCO-Seg dataset

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-seg")

model.train(data="coco.yaml", device=[0, 1, 2, 3, 4, 5, 6, 7], img_size=640, epochs=600, batch=128)
```

```bash
vajra train model=vajra-v1-nano-seg data=coco.yaml device=[0, 1, 2, 3, 4, 5, 6, 7] img_size=640 epochs=600 batch=128
```

## Citations and Acknowledgements

If you use the COCO dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).
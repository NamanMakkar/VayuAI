# COCO Pose Dataset

The [COCO-Pose](https://cocodataset.org/#keypoints-2017) dataset is a specialized version of the COCO (Common Objects in Context) dataset, designed for pose estimation tasks. It leverages the COCO Keypoints 2017 images and labels to enable the training of models like VajraV1 and YOLO for pose estimation tasks.

## COCO Pretrained Models Perf

| Model                                                                                | Size (pixels) | Pose mAP<sup>val<br>50-95</sup> | Pose mAP<sup>val<br>50</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|-------------------------------|-------------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-nano-pose.pt) | 640           | 56.4                          | 84.7                             | 1.2                                                 | 4.07       | 14.8      |
| [VajraV1-small-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-small-pose.pt) | 640           | 65                          | 88.9                             | 1.4                                                 | 12.07      | 49.6      |
| [VajraV1-medium-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-medium-pose.pt) | 640           | 68.5                          | 89.9                             | 1.8                                                 | 21.15      | 98.2      |
| [VajraV1-large-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-large-pose.pt) | 640           | 69.5                          | 90.6                             | 2.1                                                 | 25.49      | 118.9     |
| [VajraV1-xlarge-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-xlarge-pose.pt) | 640           | 71.5                          | 91.4                             | 3.7                                                 | 73.56       | 26.5      |

## Key Features

- COCO-Pose builds upon the COCO Keypoints 2017 dataset which containes 200K images labeled with keypoints for pose estimation tasks.
- The dataset supports 17 keypoints for human figures, facilitating detailed pose estimation.
- Like COCO, it provides standardized evaluation metrics, including Object Keypoint Similarity (OKS) for pose estimation tasks, making it suitable for comparing model performance.

## Dataset Structure

The COCO-Pose dataset is split into three subsets:

1. **Train2017**: This subset contains 56599 images from the COCO dataset, annotated for training pose estimation models.
2. **Val2017**: This subset has 2346 images used for validation purposes during model training.
3. **Test2017**: This subset consists of images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7384) for performance evaluation.

## Applications

The COCO-Pose dataset is specifically used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in keypoint detection and pose estimation tasks, such as OpenPose. The dataset's large number of annotated images and standardized evaluation metrics make it an essential resource for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) researchers and practitioners focused on pose estimation.

## Usage

To train a VajraV1-Pose model on the COCO-Pose dataset, use the following code snippet:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-pose")

model.train(data="coco-pose.yaml", device=[0, 1, 2, 3, 4, 5, 6, 7], batch=128, epochs=1000, img_size=640)
```

In CLI:

```bash
vajra pose train data=coco-pose.yaml model=vajra-v1-nano-pose device=[0, 1, 2, 3, 4, 5, 6, 7] batch=128 epochs=1000 img_size=640
```

## Citations and Acknowledgments

If you use the COCO-Pose dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO-Pose dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).
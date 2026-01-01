# VOC Dataset

The [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes) dataset is a well-known object detection, segmentation, and classification dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.

## Key Features

- VOC dataset includes two main challenges: VOC2007 and VOC2012.
- The dataset comprises 20 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as boats, sofas, and dining tables.
- Annotations include object bounding boxes and class labels for object detection and classification tasks, and segmentation masks for the segmentation tasks.
- VOC provides standardized evaluation metrics like mean Average Precision (mAP) for object detection and classification, making it suitable for comparing model performance.

## Dataset Structure

The VOC dataset is split into three subsets:

1. **Train**: This subset contains images for training object detection, segmentation, and classification models.
2. **Validation**: This subset has images used for validation purposes during model training.
3. **Test**: This subset consists of images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the PASCAL VOC evaluation server for performance evaluation.

## Applications

The VOC dataset is widely used for training and evaluating deep learning models in object detection (such as [VajraV1](../models/vajrav1.md), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [SSD](https://arxiv.org/abs/1512.02325)), instance segmentation (such as [Mask R-CNN](https://arxiv.org/abs/1703.06870)), and image classification. The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

## Usage

To train a VajraV1 model on the VOC dataset for 100 epochs with an image size of 640, you can use the following code snippets.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")

model.train(data="VOC.yaml", epochs=100, img_size=640)
```

```bash
vajra detect train data=VOC.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640
```

## Citations and Acknowledgments

If you use the VOC dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{everingham2010pascal,
              title={The PASCAL Visual Object Classes (VOC) Challenge},
              author={Mark Everingham and Luc Van Gool and Christopher K. I. Williams and John Winn and Andrew Zisserman},
              year={2010},
              eprint={0909.5206},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge the PASCAL VOC Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the VOC dataset and its creators, visit the [PASCAL VOC dataset website](http://host.robots.ox.ac.uk/pascal/VOC/).
# LVIS Dataset

The [LVIS dataset](https://www.lvisdataset.org/) is a large-scale, fine-grained vocabulary-level annotation dataset developed and released by Facebook AI Research (FAIR). It is primarily used as a research benchmark for object detection and instance segmentation with a large vocabulary of categories, aiming to drive further advancements in computer vision field.

## Key Features

- LVIS contains 160k images and 2M instance annotations for object detection, segmentation, and captioning tasks.
- The dataset comprises 1203 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as umbrellas, handbags, and sports equipment.
- Annotations include object bounding boxes, segmentation masks, and captions for each image.
- LVIS provides standardized evaluation metrics like mean Average Precision (mAP) for object detection, and mean Average Recall (mAR) for segmentation tasks, making it suitable for comparing model performance.
- LVIS uses exactly the same images as [COCO](./coco.md) dataset, but with different splits and different annotations.

## Dataset Structure

The LVIS dataset is split into three subsets:

1. **Train**: This subset contains 100k images for training object detection, segmentation, and captioning models.
2. **Val**: This subset has 20k images used for validation purposes during model training.
3. **Minival**: This subset is exactly the same as COCO val2017 set which has 5k images used for validation purposes during model training.
4. **Test**: This subset consists of 20k images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [LVIS evaluation server](https://eval.ai/web/challenges/challenge-page/675/overview) for performance evaluation.

## Applications

The LVIS dataset is widely used for training and evaluating deep learning models in object detection (such as [VajraV1](../models/vajrav1.md), YOLO, [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [SSD](https://arxiv.org/abs/1512.02325)), instance segmentation (such as [Mask R-CNN](https://arxiv.org/abs/1703.06870)). The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

## Usage

To train a VajraV1 model from scratch on the LVIS dataset, you can use the code snippet below:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-seg") # You can train for both segmentation and detection on the LVIS dataset, for detection use Vajra("vajra-v1-nano-det")

model.train(data="lvis.yaml", epochs=600, img_size=640, batch=128, device=[0, 1, 2, 3, 4, 5, 6, 7])
```

## Citations and Acknowledgments

If you use the LVIS dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{gupta2019lvis,
          title={LVIS: A Dataset for Large Vocabulary Instance Segmentation},
          author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
          booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
        ```

We would like to acknowledge the LVIS Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the LVIS dataset and its creators, visit the [LVIS dataset website](https://www.lvisdataset.org/).
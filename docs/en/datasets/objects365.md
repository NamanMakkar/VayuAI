# Objects365 Dataset

The [Objects365](https://www.objects365.org/) dataset is a large-scale, high-quality dataset designed to foster object detection research with a focus on diverse objects in the wild. Created by a team of [Megvii](https://en.megvii.com/) researchers, the dataset offers a wide range of high-resolution images with a comprehensive set of annotated bounding boxes covering 365 object categories.

## Key Features

- Objects365 contains 365 object categories, with 2 million images and over 30 million bounding boxes.
- The dataset includes diverse objects in various scenarios, providing a rich and challenging benchmark for object detection tasks.
- Annotations include bounding boxes for objects, making it suitable for training and evaluating object detection models.
- Objects365 pre-trained models significantly outperform ImageNet pre-trained models, leading to better generalization on various tasks.

## Dataset Structure

The Objects365 dataset is organized into a single set of images with corresponding annotations:

- **Images**: The dataset includes 2 million high-resolution images, each containing a variety of objects across 365 categories.
- **Annotations**: The images are annotated with over 30 million bounding boxes, providing comprehensive ground truth information for object detection tasks.

## Applications

The Objects365 dataset is widely used for training and evaluating deep learning models in object detection tasks. The dataset's diverse set of object categories and high-quality annotations make it a valuable resource for researchers and practitioners in the field of computer vision.

## Usage

Train a VajraV1 model on the Objects365 dataset for 100 epochs with image size of 640 by using the following code snippet.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")

model.train(data="Objects365.yaml", epochs=100, img_size=640)
```

```bash
vajra detect train data=Objects365.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640
```

## Citations and Acknowledgments

If you use the Objects365 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{shao2019objects365,
          title={Objects365: A Large-scale, High-quality Dataset for Object Detection},
          author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Li, Jing and Zhang, Xiangyu and Sun, Jian},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
          pages={8425--8434},
          year={2019}
        }
        ```

We would like to acknowledge the team of researchers who created and maintain the Objects365 dataset as a valuable resource for the computer vision research community. For more information about the Objects365 dataset and its creators, visit the [Objects365 dataset website](https://www.objects365.org/).

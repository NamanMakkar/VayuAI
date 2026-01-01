# ImageNet Dataset

[ImageNet](https://www.image-net.org/) is a large-scale database of annotated images designed for use in visual object recognition research. It contains over 14 million images, with each image annotated using WordNet synsets, making it one of the most extensive resources available for training deep learning models in computer vision.

## Key Features

- ImageNet contains over 14 million high-resolution images spanning thousands of object categories.
- The dataset is organized according to the WordNet hierarchy, with each synset representing a category.
- ImageNet is widely used for training and benchmarking in the field of computer vision, particularly for image classification and object detection tasks.
- The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) has been instrumental in advancing computer vision research.

## Dataset Structure

The ImageNet dataset is organized using the WordNet hierarchy. Each node in the hierarchy represents a category, and each category is described by a synset (a collection of synonymous terms). The images in ImageNet are annotated with one or more synsets, providing a rich resource for training models to recognize various objects and their relationships.

## ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) has been an important event in the field of computer vision. It has provided a platform for researchers and developers to evaluate their algorithms and models on a large-scale dataset with standardized evaluation metrics. The ILSVRC has led to significant advancements in the development of deep learning models for image classification, object detection, and other computer vision tasks.

## Applications

The ImageNet dataset is widely used for training and evaluating deep learning models in various computer vision tasks, such as image classification, object detection, and object localization. Some popular deep learning architectures, such as [AlexNet](https://en.wikipedia.org/wiki/AlexNet), [VGG](https://arxiv.org/abs/1409.1556), and [ResNet](https://arxiv.org/abs/1512.03385), were developed and benchmarked using the ImageNet dataset.

## Usage

### Installation

Install the VayuAI SDK and download the ImageNet dataset

```bash
git clone https://github.com/NamanMakkar/VayuAI.git
cd VayuAI/
pip install .

bash vajra/dataset/scripts/get_imagenet.sh
```

### Training

Train a VajraV1-cls model on the ImageNet dataset using the code snippet below:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-cls")

model.train(data="../data/imagenet", epochs=200, batch=256, img_size=224)
```

## Citations and Acknowledgments

If you use the ImageNet dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{ILSVRC15,
                 author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
                 title={ImageNet Large Scale Visual Recognition Challenge},
                 year={2015},
                 journal={International Journal of Computer Vision (IJCV)},
                 volume={115},
                 number={3},
                 pages={211-252}
        }
        ```

We would like to acknowledge the ImageNet team, led by Olga Russakovsky, Jia Deng, and Li Fei-Fei, for creating and maintaining the ImageNet dataset as a valuable resource for the machine learning and computer vision research community. For more information about the ImageNet dataset and its creators, visit the [ImageNet website](https://www.image-net.org/).
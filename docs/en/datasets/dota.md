# DOTA Dataset for Oriented Bounding Box Object Detection

DOTA is a special dataset for object detection in aerial images. It offers annotated images capturing a diverse array of aerial scenes with Oriented Bounding Boxes.

## Key Features

- Collection from various sensors and platforms, with image sizes ranging from 800 × 800 to 20,000 × 20,000 pixels.
- Features more than 1.7M Oriented Bounding Boxes across 18 categories.
- Encompasses multiscale object detection.
- Instances are annotated by experts using arbitrary (8 d.o.f.) quadrilateral, capturing objects of different scales, orientations, and shapes.

## Dataset Versions

### DOTA-v1.0

- Contains 15 common categories.
- Comprises 2,806 images with 188,282 instances.
- Split ratios: 1/2 for training, 1/6 for validation, and 1/3 for testing.

### DOTA-v1.5

- Incorporates the same images as DOTA-v1.0.
- Very small instances (less than 10 pixels) are also annotated.
- Addition of a new category: "container crane".
- A total of 403,318 instances.
- Released for the [DOAI Challenge 2019 on Object Detection in Aerial Images](https://captain-whu.github.io/DOAI2019/challenge.html).

### DOTA-v2.0

- Collections from Google Earth, GF-2 Satellite, and other aerial images.
- Contains 18 common categories.
- Comprises 11,268 images with a whopping 1,793,658 instances.
- New categories introduced: "airport" and "helipad".
- Image splits:
    - Training: 1,830 images with 268,627 instances.
    - Validation: 593 images with 81,048 instances.
    - Test-dev: 2,792 images with 353,346 instances.
    - Test-challenge: 6,053 images with 1,090,637 instances.


## Dataset Structure

DOTA exhibits a structured layout tailored for OBB object detection challenges:

- **Images**: A vast collection of high-resolution aerial images capturing diverse terrains and structures.
- **Oriented Bounding Boxes**: Annotations in the form of rotated rectangles encapsulating objects irrespective of their orientation, ideal for capturing objects like airplanes, ships, and buildings.

## Applications

DOTA serves as a benchmark for training and evaluating models specifically tailored for aerial image analysis. With the inclusion of OBB annotations, it provides a unique challenge, enabling the development of specialized object detection models that cater to aerial imagery's nuances. The dataset is particularly valuable for applications in remote sensing, surveillance, and environmental monitoring.

## Usage

Splitting the DOTAv1 dataset by using multiple patches from each image to create a larger multi-scale dataset is necessary for optimal mAP. The code snippet below creates a dataset consisting of 1024 x 1024 images using patches from the original DOTAv1 dataset images. This is done as the original images of the DOTAv1 dataset are extemely large, it is necessary to create a larger dataset from the patches of each image.

```python
from vajra.dataset.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="/root/data/DOTAv1/",
    save_dir="/root/data/DOTAv1-split/",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)
# split test set, without labels.
split_test(
    data_root="/root/data/DOTAv1/",
    save_dir="/root/data/DOTAv1-split/",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)
```

Use the following code snippet to train on the DOTAv1 dataset:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det")

model.train(data="DOTAv1.yaml", device=0, img_size=1024, batch=32, mixup=0.1, epochs=200)
```

## Citations and Acknowledgements

If you use the DOTA dataset in your research, it's important to cite the relevant research papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

A special note of gratitude to the team behind the DOTA datasets for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DOTA website](https://captain-whu.github.io/DOTA/index.html).
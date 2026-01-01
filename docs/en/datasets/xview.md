# xView Dataset

The [xView](http://xviewdataset.org/) dataset is one of the largest publicly available datasets of overhead imagery, containing images from complex scenes around the world annotated using bounding boxes. The goal of the xView dataset is to accelerate progress in four computer vision frontiers:

1. Reduce minimum resolution for detection.
2. Improve learning efficiency.
3. Enable discovery of more object classes.
4. Improve detection of fine-grained classes.

!!! warning "Manual Download Required"

    The xView dataset is **not** automatically downloaded by VayuAI scripts. You **must** manually download the dataset first from the official source:

    - **Source:** DIUx xView 2018 Challenge by U.S. National Geospatial-Intelligence Agency (NGA)
    - **URL:** [https://challenge.xviewdataset.org](https://challenge.xviewdataset.org)

    **Important:** After downloading the necessary files (e.g., `train_images.tif`, `val_images.tif`, `xView_train.geojson`), you need to extract them and place them into the correct directory structure, typically expected under a `datasets/xView/` folder, **before** running the training commands provided below. Ensure the dataset is properly set up as per the challenge instructions.

## Key Features

- xView contains over 1 million object instances across 60 classes.
- The dataset has a resolution of 0.3 meters, providing higher resolution imagery than most public satellite imagery datasets.
- xView features a diverse collection of small, rare, fine-grained, and multi-type objects with bounding box annotation.
- Comes with a pre-trained baseline model using the TensorFlow object detection API and an example for PyTorch.

## Dataset Structure

The xView dataset is composed of satellite images collected from WorldView-3 satellites at a 0.3m ground sample distance. It contains over 1 million objects across 60 classes in over 1,400 km² of imagery. The dataset is particularly valuable for remote sensing applications.

## Applications

The xView dataset is widely used for training and evaluating deep learning models for object detection in overhead imagery. The dataset's diverse set of object classes and high-resolution imagery make it a valuable resource for researchers and practitioners in the field of computer vision, especially for satellite imagery analysis. Applications include:

- Military and defense reconnaissance
- Urban planning and development
- Environmental monitoring
- Disaster response and assessment
- Infrastructure mapping and management

## Usage

To train a VajraV1 model on the xView dataset for 100 epochs with an image size of 640, you can use the following code snippets.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")

model.train(data="xView.yaml", epochs=100, img_size=640)
```

```bash
vajra detect train data=xView.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640
```

## Related Datasets

If you're working with satellite imagery, you might also be interested in exploring these related datasets:

- [DOTA-v2](../datasets/dota.md): A dataset for oriented object detection in aerial images
- [VisDrone](../datasets/visdrone.md): A dataset for object detection and tracking in drone-captured imagery
- [Argoverse](../datasets/argoverse.md): A dataset for autonomous driving with 3D tracking annotations

## Citations and Acknowledgments

If you use the xView dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
              title={xView: Objects in Context in Overhead Imagery},
              author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
              year={2018},
              eprint={1802.07856},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```
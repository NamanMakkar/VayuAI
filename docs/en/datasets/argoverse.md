# Argoverse Dataset

The [Argoverse](https://www.argoverse.org/) dataset is a collection of data designed to support research in autonomous driving tasks, such as 3D tracking, motion forecasting, and stereo depth estimation. Developed by Argo AI, the dataset provides a wide range of high-quality sensor data, including high-resolution images, LiDAR point clouds, and map data.

## Key Features

- Argoverse contains over 290K labeled 3D object tracks and 5 million object instances across 1,263 distinct scenes.
- The dataset includes high-resolution camera images, LiDAR point clouds, and richly annotated HD maps.
- Annotations include 3D bounding boxes for objects, object tracks, and trajectory information.
- Argoverse provides multiple subsets for different tasks, such as 3D tracking, motion forecasting, and stereo depth estimation.

## Dataset Structure

The Argoverse dataset is organized into three main subsets:

1. **Argoverse 3D Tracking**: This subset contains 113 scenes with over 290K labeled 3D object tracks, focusing on 3D object tracking tasks. It includes LiDAR point clouds, camera images, and sensor calibration information.
2. **Argoverse Motion Forecasting**: This subset consists of 324K vehicle trajectories collected from 60 hours of driving data, suitable for motion forecasting tasks.
3. **Argoverse Stereo Depth Estimation**: This subset is designed for stereo depth estimation tasks and includes over 10K stereo image pairs with corresponding LiDAR point clouds for ground truth depth estimation.

## Applications

The Argoverse dataset is widely used for training and evaluating deep learning models in autonomous driving tasks such as 3D object tracking, motion forecasting, and stereo depth estimation. The dataset's diverse set of sensor data, object annotations, and map information make it a valuable resource for researchers and practitioners in the field of autonomous driving.

## Usage

To train a VajraV1 model on the Argoverse dataset for 100 epochs with an image size of 640, you can use the following code snippets.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.train(data="Argoverse.yaml", epochs=100, img_size=640)
```

```bash
vajra detect train model=vajra-v1-nano-det.pt data=Argoverse.yaml epochs=100 img_size=640
```

## Citations and Acknowledgments

If you use the Argoverse dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{chang2019argoverse,
          title={Argoverse: 3D Tracking and Forecasting with Rich Maps},
          author={Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, Dequan and Carr, Peter and Lucey, Simon and Ramanan, Deva and others},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={8748--8757},
          year={2019}
        }
        ```

We would like to acknowledge Argo AI for creating and maintaining the Argoverse dataset as a valuable resource for the autonomous driving research community. For more information about the Argoverse dataset and its creators, visit the [Argoverse dataset website](https://www.argoverse.org/).
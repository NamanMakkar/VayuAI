# VisDrone Dataset

The [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) is a benchmark created by the AISKYEYE team at the Lab of Machine Learning and Data Mining, Tianjin University, China. It contains carefully annotated ground truth data for various computer vision tasks related to drone-based image and video analysis.

VisDrone is composed of 288 video clips with 261,908 frames and 10,209 static images, captured by various drone-mounted cameras. The dataset covers a wide range of aspects, including location (14 different cities across China), environment (urban and rural), objects (pedestrians, vehicles, bicycles, etc.), and density (sparse and crowded scenes). The dataset was collected using various drone platforms under different scenarios and weather and lighting conditions. These frames are manually annotated with over 2.6 million bounding boxes of targets such as pedestrians, cars, bicycles, and tricycles. Attributes like scene visibility, object class, and occlusion are also provided for better data utilization.

## Dataset Structure

The VisDrone dataset is organized into five main subsets, each focusing on a specific task:

1. **Task 1**: Object detection in images
2. **Task 2**: Object detection in videos
3. **Task 3**: Single-object tracking
4. **Task 4**: [Multi-object tracking](../datasets/index.md#multi-object-tracking)
5. **Task 5**: Crowd counting

## Applications

The VisDrone dataset is widely used for training and evaluating deep learning models in drone-based computer vision tasks such as object detection, object tracking, and crowd counting. The dataset's diverse set of sensor data, object annotations, and attributes make it a valuable resource for researchers and practitioners in the field of drone-based computer vision.

## Usage

Train a VajraV1 model on the VisDrone Dataset using the code snippet below:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det") # Training from scratch, use Vajra("vajra-v1-nano-det.pt") for COCO pretrained weights

model.train(data="VisDrone.yaml", device=0, batch=12, epochs=500, img_size=640)
```

```bash
vajra train model=vajra-v1-nano-det device=0 batch=12 epochs=500 img_size=640
```

## Citations and Acknowledgement

If you use the VisDrone dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}}
        ```

We would like to acknowledge the AISKYEYE team at the Lab of Machine Learning and [Data Mining](https://www.ultralytics.com/glossary/data-mining), Tianjin University, China, for creating and maintaining the VisDrone dataset as a valuable resource for the drone-based computer vision research community. For more information about the VisDrone dataset and its creators, visit the [VisDrone Dataset GitHub repository](https://github.com/VisDrone/VisDrone-Dataset).
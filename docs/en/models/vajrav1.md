# Vayuvahana Technologies VajraV1

Vayuvahana Technologies Private Limited [VajraV1](https://github.com/NamanMakkar/VayuAI) is a 
state-of-the-art (SOTA) real time object detection model inspired by the YOLO model architectures. VajraV1 is a family of fast, lightweight models that can be used for a variety of tasks like object detection and tracking, instance segmentation, oriented object detection, pose detection, and image classification.

## VajraV1 Object Detection Performance on the COCO Dataset

| Model                                                                                | Size (pixels) | mAP<sup>val<br>50-95</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|----------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-nano-det.pt) | 640           | 44.3                       | 1.0                                                 | 3.78       | 13.7       |
| [VajraV1-small-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-small-det.pt)  | 640           | 50.4                       | 1.1                                                 | 11.58      | 47.9      |
| [VajraV1-medium-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-medium-det.pt)  | 640           | 52.7                           | 1.5                                                 | 20.29      | 94.5      |
| [VajraV1-large-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-large-det.pt)  | 640           | 53.7                           | 1.8                                                 | 24.63      | 115.2      |
| [VajraV1-xlarge-det](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-xlarge-det.pt)  | 640           | 56.2                           | 3.2                                                 | 72.7      | 208.3     |

<p align="center">
<img width="100%" src="./vajra/assets/vajra_v1_detection_performance_coco.png" alt="VajraV1 Detection Performance on COCO Dataset">
</p>

## VajraV1 Instance Segmentation Performance on the COCO Dataset

| Model                                                                                | Size (pixels) | Box mAP<sup>val<br>50-95</sup> | Mask mAP<sup>val<br>50-95</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|-------------------------------|-------------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-nano-seg.pt) | 640           | 43.6                          | 35.8                             | 1.2                                                 | 4.03       | 17.6      |
| [VajraV1-small-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-small-seg.pt) | 640           | 50.2                          | 40.5                             | 1.2                                                 | 12.23      | 61.9      |
| [VajraV1-medium-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-medium-seg.pt) | 640           | 52.6                          | 42.3                             | 1.7                                                 | 22.6      | 149.9      |
| [VajraV1-large-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-large-seg.pt) | 640           | 53.6                          | 43.1                             | 2.0                                                 | 26.93      | 170.6     |
| [VajraV1-xlarge-seg](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-xlarge-seg.pt) | 640           | 55.7                          | 44.5                             | 3.4                                                 | 75       | 278.1     |

<p align="center">
<img width="100%" src="./vajra/assets/vajra_v1_segmentation_performance_coco.png" alt="VajraV1 Segmentation Performance on COCO Dataset">
</p>

## VajraV1 Pose Estimation Performance on the COCO Dataset

| Model                                                                                | Size (pixels) | Pose mAP<sup>val<br>50-95</sup> | Pose mAP<sup>val<br>50</sup> | Speed<br><sup>RTX 4090 TensorRT10 Latency (ms)</sup> | Params (M) | FLOPs (B) |
|-------------------------------------------------------------------------------------|---------------|-------------------------------|-------------------------------|-----------------------------------------------------|------------|-----------|
| [VajraV1-nano-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-nano-pose.pt) | 640           | 56.4                          | 84.7                             | 1.2                                                 | 4.07       | 14.8      |
| [VajraV1-small-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-small-pose.pt) | 640           | 65                          | 88.9                             | 1.4                                                 | 12.07      | 49.6      |
| [VajraV1-medium-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-medium-pose.pt) | 640           | 68.5                          | 89.9                             | 1.8                                                 | 21.15      | 98.2      |
| [VajraV1-large-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-large-pose.pt) | 640           | 69.5                          | 90.6                             | 2.1                                                 | 25.49      | 118.9     |
| [VajraV1-xlarge-pose](https://github.com/NamanMakkar/VayuAI/releases/download/v1.0.4/vajra-v1-xlarge-pose.pt) | 640           | 71.5                          | 91.4                             | 3.7                                                 | 73.56       | 26.5      |

<p align="center">
<img width="100%" src="./vajra/assets/vajra_v1_pose_performance.png" alt="VajraV1 Pose Estimation Performance on COCO Dataset">
</p>


# VajraV1 vs YOLO

The VajraV1 model is an iterative improvement over the YOLOv9, YOLOv10, YOLO11 and YOLO12. It uses techniques inspired by the YOLOv9 and YOLOv10 models to optimize the model architecture of the backbone and the neck of the model and utilizes the anchor free detection head of the YOLO11 to improve upon the existing YOLO models. 

The VajraV1 model architecture optimizes the primary computational blocks uses in the backbone and the neck of the network and efficiently integrates self-attention into the backbone of the object detector. The VajraV1 models achieve significantly better Mean Average Precision (mAP) than their YOLO counterparts while achieving competitive latency.

## Performance Comparison

We provide a comparison of the VajraV1 models with the YOLO models on different tasks supported by the VayuAI SDK.

### Object Detection

<p align="center">
<img width="100%" src="../../../vajra/assets/vajra_v1_detection_performance_coco.png" alt="VajraV1 Detection Performance on COCO Dataset">
</p>

### Object Instance Segmentation

<p align="center">
<img width="100%" src="../../../vajra/assets/vajra_v1_segmentation_performance_coco.png" alt="VajraV1 Segmentation Performance on COCO Dataset">
</p>

### Pose Estimation

<p align="center">
<img width="100%" src="../../../vajra/assets/vajra_v1_pose_performance.png" alt="VajraV1 Pose Estimation Performance on COCO Dataset">
</p>
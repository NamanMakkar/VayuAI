# Datasets Overview

This guide introduces you to the various datasets and dataset formats supported by the VayuAI SDK and provides insights into their structure, usage, and how to convert between different formats. 

## Supported Datasets

- [Argoverse](argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations.
- [Brain-tumor](brain-tumor.md): A dataset for detecting brain tumors includes MRI or CT scan images with details on tumor presence, location, and characteristics.
- [COCO](coco.md): Common Objects in Context (COCO) is a large-scale object detection, segmentation, and captioning dataset with 80 object categories.
- [LVIS](lvis.md): A large-scale object detection, segmentation, and captioning dataset with 1203 object categories.
- [DOTA-v1](dota.md): The first version of the DOTA dataset, providing a comprehensive set of aerial images with oriented bounding boxes for object detection.
- [DOTA-v1.5](dota.md): An intermediate version of the DOTA dataset, offering additional annotations and improvements over DOTA-v1 for enhanced object detection tasks.
- [DOTA-v2](dota.md): DOTA (A Large-scale Dataset for Object Detection in Aerial Images) version 2, emphasizes detection from aerial perspectives and contains oriented bounding boxes with 1.7 million instances and 11,268 images.
- [ImageNet](imagenet.md): A large-scale dataset for object detection and image classification with over 14 million images and 20,000 categories.
- [Objects365](objects365.md): A high-quality, large-scale dataset for object detection with 365 object categories and over 600K annotated images.
- [OpenImagesV7](open-images-v7.md): A comprehensive dataset by Google with 1.7M train images and 42k validation images.
- [SARDet-100k](sardet_100k.md): A large scale object detection dataset containing synthetic aperture radar images. 94k train images, 10k validation images and 11k test images.
- [VisDrone](visdrone.md): A dataset containing object detection and multi-object tracking data from drone-captured imagery with over 10K images and video sequences.
- [VOC](voc.md): The Pascal Visual Object Classes (VOC) dataset for object detection and segmentation with 20 object classes and over 11K images.
- [xView](xview.md): A dataset for object detection in overhead imagery with 60 object categories and over 1 million annotated objects.

## Port or Convert Label Formats

### COCO Dataset Format to YOLO Format

You can easily convert labels from the popular [COCO dataset](coco.md) format to the YOLO format using the following code snippet:

!!! example

    === "Python"

        ```python
        from vajra.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/")
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format. The process transforms the JSON-based COCO annotations into the simpler text-based YOLO format, making it compatible with [VajraV1 models](../models/vajrav1.md).

Remember to double-check if the dataset you want to use is compatible with your model and follows the necessary format conventions. Properly formatted datasets are crucial for training successful object detection models.

## OBB Dataset Format

The OBB Dataset format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1. It follows this format:

```bash
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Internally, outputs are processed in the `xywhr` format which represents the bounding box center point point (xy), width, height and rotation.

An example of a `*.txt` label file for the above image, which contains an object of class `0` in OBB format, could look like:

```bash
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
```

## Segment Dataset Format

This is the format of the segmentation datasets used in the VayuAI SDK for training the VajraV1 models:

1. One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2. One row per object: Each row in the text file corresponds to one object instance in the image.
3. Object information per row: Each row contains the following information about the object instance:
    - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
    - Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.

The format for a single row in the segmentation dataset file is as follows:

```
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

In this format, `<class-index>` is the index of the class for the object, and `<x1> <y1> <x2> <y2> ... <xn> <yn>` are the bounding coordinates of the object's segmentation mask. The coordinates are separated by spaces.

Example of the dataset format for a single image with 2 objects made up of a 3-point segmentation mask and a 5-point segmentation mask:

```
0 0.681 0.485 0.670 0.487 0.676 0.487
1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104
```

!!! tip

      - The length of each row does **not** have to be equal.
      - Each segmentation label must have a **minimum of 3 xy points**: `<class-index> <x1> <y1> <x2> <y2> <x3> <y3>`

## Pose Dataset Format

Each row of the txt labels file contains the following information about the object instance:

- Object class index: An integer representing the class of the object (e.g 0 for person, 1 for car etc.).
- Object center coordinates: The x and y coordinates of the center of the object, normalized to be between 0 and 1.
- Object width and height: The width and height of the object, normalized to be between 0 and 1.
- Object keypoint coordinates: The keypoints of the object, normalized to be between 0 and 1.

Here is an example of the label format for pose estimation:

Format with Dim=2

```
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

Format with Dim = 3

```
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <pn-visibility>
```

In this format, `<class-index>` is the index of the class for the object,`<x> <y> <width> <height>` are coordinates of bounding box, and `<px1> <py1> <px2> <py2> ... <pxn> <pyn>` are the pixel coordinates of the keypoints. The coordinates are separated by spaces.

## Classification Dataset Format

For classification, the dataset must be organized in a specific split-directory structure under the `root` directory to facilitate proper training, testing and optional validation processes. This structure includes separate directories for training (`train`) and testing (`test`) phases, with an optional directory for validation (`val`).

Each of these directories should contain one subdirectory for each class in the dataset. The subdirectories are named after the corresponding class and contain all the images for that class. Ensure that each image file is named uniquely and stored in a common format such as JPEG or PNG.

### Folder Structure Example

Consider the [ImageNet](imagenet.md) dataset as an example. The folder structure should look like this:

```
imagenet/
├── train/
│   ├── nXXXXXXXX/
│   └── ... (1000 folders)
└── val/
    ├── nXXXXXXXX/
    └── ... (1000 folders, 50 imgs each)
```

## Auto Annotation for generating segmentation annotations

You can use the combination of the VajraV1 models and the Segment Anything Models to generate segmentation annotations for your dataset. Use this if you do not have detection annotations for your data. 

```python
from vajra.dataset.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="vajra-v1-xlarge-det.pt", sam_model="sam2_l.pt")
```

Alternatively, if you do have detection annotations and want to generate segmentation annotations, use the Segment Anything Model to generate segmentation annotations.

```python
from vajra.dataset.converter import yolo_bbox2segment

yolo_bbox2segment(im_dir="path/to/images", save_dir="path/to/dir", sam_model="sam2_l.pt", device=0)
```
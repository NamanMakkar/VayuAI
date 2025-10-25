# Computer Vision Tasks supported by Vayuvahana Technologies VajraV1

## [Detection](detect.md)

Detection is the primary task supported by VajraV1. It involves identifying objects in an image or video frame and drawing bounding boxes around them, The detected objects are classified into different categories based on their features. VajraV1 can detect multiple objects in a single image or video frame with high accuracy and speed, making it ideal for real-time applications like surveillance systems and autonomous vehicles.

## [Image segmentation](segment.md)

Segmentation further builds upon object detection by segmenting an image into different regions based on content. Each region is assigned a label, providing pixel-level precision for applications such as medical imaging, agricultural analysis and manufacturing quality control. VajraV1-seg models implement a variant of the U-Net architecture to perform efficient and accurate segmentation.

## [Classification](classify.md)

Classification involves categorizing entire images based on their content. VajraV1's classification capabilities leverage the backbone of the VajraV1 object detector to deliver high performance image classification. This task is essential for applications like product categorization and disease classification in medical imaging.

## Multilabel Classification

Multilabel classification takes the classification task a step further by allowing more than one label for each image. This task is essential in medical imaging where multiple abnormalities need to be detected in an image and the image cannot be put into a single category. For example, in the Chest14 dataset each image has multiple lung abnormalities and therefore multiple labels have to be assigned to each image.

## [Pose estimation](pose.md)

Pose estimation detects specific keypoints in images or video frames to track movements or estimate poses. These keypoints can represent human joints, facial features, or other significant points of interest. VajraV1 excels at keypoint detection with high accuracy and speed, making it valuable for fitness applications, sports analytics, action recognition.

## [OBB](obb.md)

Oriented Bounding Box (OBB) detection enhances traditional object detection by adding an orientation angle better locate rotated objects. VajraV1 does this by using an additional parameter in the detection head which measures the angle or orientation of the bounding boxes. This is valuable for aerial imagery analysis and industrial application where objects appear at various angles. VajraV1 offers high accuracy and speed for detecting rotated objects in diverse scenarious.

## [Small Object Detection](small_obj_detect.md)

Small Object Detection is a novel technique for enhancing object detection. It uses a pose estimation model to locate the centroid of a bounding box while the detection branch is trained to estimate the 4 corners of the bounding box. This helps enhance object detection for small objects as it estimates both the centroid of the bounding box as well as the bounding box. For training a pose model is utilised and the weights are then transfered to a detection model for validation. This offers improved detection of smaller objects in overhead imagery with absolutely no latency overhead compared to the original detection models.

## Conclusion

Vayuvahana Technologies VajraV1 supports multiple computer vision tasks, including detection, segmentation, pose estimation, classification, multilabel classification, oriented object detection, small object detection. Each task addresses specific needs in the computer vision landscape, from basic object identification to detailed pose analysis. By understanding the capabilities and applications of each task, you can select the most appropriate approach for your specific computer vision problems and leverage VajraV1's powerful features to build effective solutions.


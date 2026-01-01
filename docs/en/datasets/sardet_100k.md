# SARDet-100K Dataset

The SARDet-100K Dataset is a large-scale benchmark for Synthetic Aperture Radar (SAR) object detection, introduced in the NeurIPS 2024 Spotlight paper "SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection". It was created by researchers from ShanghaiTech University, Northwestern Polytechnical University, and other institutions to address the scarcity of diverse, large-scale public SAR datasets.

SARDet-100K is the first COCO-scale multi-class SAR object detection dataset, comprising 116,598 images with 245,653 annotated instances across six categories: Ship, Aircraft, Car, Bridge, Tank, and Harbor. The dataset was built by surveying, collecting, and standardizing annotations from 10 existing SAR datasets, ensuring diversity in sensors (e.g., Gaofen-3, Sentinel-1), polarizations, resolutions, and scenarios (inshore/offshore ships, airports, vehicles, etc.). Images are provided in standard formats (horizontal bounding boxes in COCO-style JSON), with many patches cropped to 512×512 or 800×800 for training efficiency. This diversity and scale make it ideal for training robust SAR-specific models under varying imaging conditions.

## Dataset Structure

SARDet-100K follows a COCO-like structure with train/val/test splits:

Train: 94,493 images
Val: 10,492 images
Test: 11,613 images
Annotations are in COCO JSON format (horizontal bounding boxes). A rotated bounding box variant is also provided in a companion release for oriented object detection tasks.

The dataset primarily supports:

Object Detection (horizontal/rotated bounding boxes in SAR images)

It serves as a unified benchmark, replacing smaller, mono-category SAR datasets.

## Applications

SARDet-100K is designed for training and evaluating deep learning models in SAR-based object detection, particularly for all-weather, day-night remote sensing applications such as maritime surveillance, military reconnaissance, disaster monitoring, and airport/vehicle tracking. Its large scale and multi-class nature enable better generalization than previous small SAR datasets, supporting advancements in domain adaptation from optical to SAR imagery.

## Usage

Train a VajraV1 model on the SARDet-100K Dataset using the code snippet below:

```python
from vajra import Vajra
model = Vajra("vajra-v1-nano-det")  # Training from scratch, use Vajra("vajra-v1-nano-det.pt") for COCO pretrained weights
model.train(data="sardet_100k.yaml", device=[0, 1, 2, 3, 4, 5, 6, 7], batch=128, epochs=600, img_size=640, degrees=180)
```

```bash
vajra train model=vajra-v1-nano-det device=[0, 1, 2, 3, 4, 5, 6, 7] batch=128 epochs=600 img_size=640 data=sardet_100k.yaml degrees=180
```

## Citations and Acknowledgement

If you use the SARDet-100K dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{li2024sardet100k,
            title={SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection},
            author={Li, Yuxuan and Li, Xiang and Li, Weijie and Hou, Qibin and Liu, Li and Wang, Ming-Hsuan and Lu, Huimin},
            booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
            year={2024}
        }
        ```

We would like to acknowledge the authors (Yuxuan Li et al.) and contributing researchers (Bo Zhang, Chenglong Li, Tian Tian, Tianwen Zhang, Xiaoling Zhang, and others) for creating and open-sourcing SARDet-100K as a foundational resource for the SAR object detection community. For more information, dataset download links, and the official toolkit, visit the [SARDet-100K GitHub repository](https://github.com/zcablii/SARDet_100K).

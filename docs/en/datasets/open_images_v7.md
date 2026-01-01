# Open Images V7 Dataset

[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) is a versatile and expansive dataset championed by Google. Aimed at propelling research in the realm of computer vision, it boasts a vast collection of images annotated with a plethora of data, including image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives.

## Key Features

- Encompasses ~9M images annotated in various ways to suit multiple computer vision tasks.
- Houses a staggering 16M bounding boxes across 600 object classes in 1.9M images. These boxes are primarily hand-drawn by experts ensuring high precision.
- Visual relationship annotations totaling 3.3M are available, detailing 1,466 unique relationship triplets, object properties, and human activities.
- V5 introduced segmentation masks for 2.8M objects across 350 classes.
- V6 introduced 675k localized narratives that amalgamate voice, text, and mouse traces highlighting described objects.
- V7 introduced 66.4M point-level labels on 1.4M images, spanning 5,827 classes.
- Encompasses 61.4M image-level labels across a diverse set of 20,638 classes.
- Provides a unified platform for image classification, object detection, relationship detection, instance segmentation, and multimodal image descriptions.

## Dataset Structure

Open Images V7 is structured in multiple components catering to varied computer vision challenges:

- **Images**: About 9 million images, often showcasing intricate scenes with an average of 8.3 objects per image.
- **Bounding Boxes**: Over 16 million boxes that demarcate objects across 600 categories.
- **Segmentation Masks**: These detail the exact boundary of 2.8M objects across 350 classes.
- **Visual Relationships**: 3.3M annotations indicating object relationships, properties, and actions.
- **Localized Narratives**: 675k descriptions combining voice, text, and mouse traces.
- **Point-Level Labels**: 66.4M labels across 1.4M images, suitable for zero/few-shot semantic segmentation.

## Applications

Open Images V7 is a cornerstone for training and evaluating state-of-the-art models in various computer vision tasks. The dataset's broad scope and high-quality annotations make it indispensable for researchers and developers specializing in computer vision.

Some key applications include:

- **Advanced Object Detection**: Train models to identify and locate multiple objects in complex scenes with high accuracy.
- **Semantic Understanding**: Develop systems that comprehend visual relationships between objects.
- **Image Segmentation**: Create precise pixel-level masks for objects, enabling detailed scene analysis.
- **Multi-modal Learning**: Combine visual data with text descriptions for richer AI understanding.
- **Zero-shot Learning**: Leverage the extensive class coverage to identify objects not seen during training.

## Usage

To train the VajraV1 model on the Open Images V7 dataset for 100 epochs with an image size of 640, you can use the following code snippets. 

!!! warning

    The complete Open Images V7 dataset comprises 1,743,042 training images and 41,620 validation images, requiring approximately **561 GB of storage space** upon download.

    Executing the commands provided below will trigger an automatic download of the full dataset if it's not already present locally. Before running the below example it's crucial to:

    - Verify that your device has enough storage capacity.
    - Ensure a robust and speedy internet connection.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")

model.train(data="open-images-v7.yaml", epochs=100, img_size=640, device=[0, 1, 2, 3, 4, 5, 6, 7], batch=128)
```

```bash
vajra detect train model=vajra-v1-nano-det.pt data=open-images-v7.yaml epochs=100 img_size=640 batch=128 device=[0, 1, 2, 3, 4, 5, 6, 7]
```

## Citations and Acknowledgments

For those employing Open Images V7 in their work, it's prudent to cite the relevant papers and acknowledge the creators:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{OpenImages,
          author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
          title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
          year = {2020},
          journal = {IJCV}
        }
        ```

A heartfelt acknowledgment goes out to the Google AI team for creating and maintaining the Open Images V7 dataset. For a deep dive into the dataset and its offerings, navigate to the [official Open Images V7 website](https://storage.googleapis.com/openimages/web/index.html).
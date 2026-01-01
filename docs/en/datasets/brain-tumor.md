# Brain Tumor Dataset

A brain tumor detection dataset consists of medical images from MRI or CT scans, containing information about brain tumor presence, location, and characteristics. This dataset is essential for training computer vision algorithms to automate brain tumor identification, aiding in early diagnosis and treatment planning in healthcare applications.

## Dataset Structure

The brain tumor dataset is divided into two subsets:

- **Training set**: Consisting of 893 images, each accompanied by corresponding annotations.
- **Testing set**: Comprising 223 images, with annotations paired for each one.

The dataset contains two classes:

- **Negative**: Images without brain tumors
- **Positive**: Images with brain tumors

## Applications

The application of brain tumor detection using computer vision enables early diagnosis, treatment planning, and monitoring of tumor progression. By analyzing medical imaging data like MRI or CT scans, computer vision systems assist in accurately identifying brain tumors, aiding in timely medical intervention and personalized treatment strategies.

Medical professionals can leverage this technology to:

- Reduce diagnostic time and improve accuracy
- Assist in surgical planning by precisely locating tumors
- Monitor treatment effectiveness over time
- Support research in oncology and neurology

## Usage

To train a VajraV1 model on the brain tumor dataset for 100 epochs with an image size of 640, utilize the provided code snippets.

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.train(data="brain-tumor.yaml", epochs=100, img_size=640)
```

```bash
vajra detect train data=brain-tumor.yaml model=vajra-v1-nano-det.pt epochs=100 img_size=640
```

## Citations and Acknowledgments

The dataset has been made available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by Ultralytics LLC.

If you use this dataset in your research or development work, please cite it appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Ultralytics_Brain_Tumor_Dataset_2023,
            author = {Ultralytics},
            title = {Brain Tumor Detection Dataset},
            year = {2023},
            publisher = {Ultralytics},
            url = {https://docs.ultralytics.com/datasets/detect/brain-tumor/}
        }
        ```

We would like to acknowledge the Ultralytics team for maintaining this dataset as a valuable resource for the computer vision community.
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import json
import asyncio
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from vajra.utils import LOGGER, TQDM, DATASETS_DIR, yaml_save
from vajra.checks import check_requirements, check_file
from vajra.utils.files import increment_path

def coco91_to_coco80_class():
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]

def coco80_to_coco91_class():
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

def convert_coco(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,):

    save_dir = increment_path(save_dir)
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)

    coco80 = coco91_to_coco80_class()

    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        images = {f'{x["id"]:d}':x for x in data["images"]}
        image_to_anns = defaultdict(list)

        for ann in data["annotations"]:
            image_to_anns[ann["image_id"]].append(ann)
        
        for img_id, anns in TQDM(image_to_anns.items(), desc=f'Annotations {json_file}'):
            img = images[f"{img_id:d}"]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            segments = []
            keypoints = []

            for ann in anns:
                if ann['iscrowd']:
                    continue
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue

            cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
                if use_segments and ann.get("segmentation") is not None:
                    if len(ann["segmentation"]) == 0:
                        segments.append([])
                        continue
                    elif len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h]).reshape(-1).tolist())
                    else:
                        s = [j for i in ann["segmentation"] for j in i]
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h]).reshape(-1).tolist())
                    s = [cls] + s
                    segments.append(s)
                if use_keypoints and ann.get("keypoints") is not None:
                    keypoints.append(
                        box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                    )
        with open((fn / f).with_suffix(".txt"), "a") as file:
            for i in range(len(bboxes)):
                if use_keypoints:
                    line = (*(keypoints[i]),)
                else:
                    line = (
                        *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                    )
                file.write(("%g " * len(line)).rstrip() % line + "\n")
    
    LOGGER.info(f'COCO data converted successfully.\nResults saved to {save_dir.resolve()}')

def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Converts DOTA dataset annotations to Vajra OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in Vajra OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from vajra.dataset.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb('path/to/DOTA')
        ```

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # Class names to indices mapping
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to Vajra OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt", device=None):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB)
    in Vajra format. Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None. Default: None.
        sam_model (str): Segmentation model to use for intermediate segmentation data; optional.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ..
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ..
                └─ NNN.txt
    """
    from vajra.dataset import VajraDetDataset
    from vajra.ops import xywh2xyxy
    from vajra.utils import LOGGER
    from vajra import SAM
    from tqdm import tqdm

    # NOTE: add placeholder to pass class index check
    dataset = VajraDetDataset(im_dir, data=dict(names=list(range(1000))))
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for l in tqdm(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = l["shape"]
        boxes = l["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(l["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        l["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for l in dataset.labels:
        texts = []
        lb_name = Path(l["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = l["cls"]
        for i, s in enumerate(l["segments"]):
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        if texts:
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")

async def convert_ndjson_to_yolo(ndjson_path: str | Path, output_path: str | Path | None = None) -> Path:
    check_requirements("aiohttp")
    import aiohttp

    ndjson_path = Path(check_file(ndjson_path))
    output_path = Path(output_path or DATASETS_DIR)
    with open(ndjson_path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    dataset_record, image_records = lines[0], lines[1:]
    dataset_dir = output_path / ndjson_path.stem
    splits = {record["split"] for record in image_records}

    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = dict(dataset_record)
    data_yaml["names"] = {int(k): v for k, v in dataset_record.get("class_names", {}).items()}
    data_yaml.pop("class_names")

    for split in sorted(splits):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        data_yaml[split] = f"images/{split}"

    async def process_record(session, semaphore, record):
        async with semaphore:
            split, original_name=  record["split"], record["file"]
            label_path = dataset_dir / "labels" / split / f"{Path(original_name).stem}.txt"
            image_path = dataset_dir / "images" / split / original_name

            annotations = record.get("annotations", {})
            lines_to_write = []
            for key in annotations.keys():
                lines_to_write = [" ".join(map(str, item)) for item in annotations[key]]
                break
            if "classification" in annotations:
                lines_to_write = [str(cls) for cls in annotations["classification"]]

            label_path.write_text("\n".join(lines_to_write) + "\n" if lines_to_write else "")

            if http_url := record.get("url"):
                if not image_path.exists():
                    try:
                        async with session.get(http_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            response.raise_for_status()
                            with open(image_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                        return True
                    except Exception as e:
                        LOGGER.warning(f"Failed to download {http_url}: {e}")
                        return False
            return True
        
    semaphore = asyncio.Semaphore(64)

    async with aiohttp.ClientSession() as session:
        pbar = TQDM(
            total=len(image_records),
            desc=f"Converting {ndjson_path.name} -> {dataset_dir} ({len(image_records)} images)",
        )

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result
        
        await asyncio.gather(*[tracked_process(record) for record in image_records])
        pbar.close()

    yaml_path = dataset_dir / "data.yaml"
    yaml_save(yaml_path, data_yaml)

    return yaml_path
            

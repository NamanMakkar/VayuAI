# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import json
import contextlib
from typing import Any
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset

from vajra.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from vajra.ops import resample_segments, segments_to_boxes
from .converter import merge_multi_segment
from .augment import Compose, Format, Instances2d, LetterBox, RandomLoadText, classify_augmentations, classify_transforms, vajra_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label

DATASET_CACHE_VERSION = "1.0.1"

def load_dataset_cache_file(path):
    """Load a *.cache dictionary from path."""
    import gc

    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache

def save_dataset_cache_file(prefix, path, x):
    """Save a dataset *.cache dictionary x to path."""
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING! Cache directory {path.parent} is not writeable, cache not saved.")

class VajraDetDataset(BaseDataset):
    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the Dataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose" or task == "small_obj_detect"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for Vajra training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING! Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING! No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = vajra_transforms(self, self.img_size, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.img_size, self.img_size), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0 
        self.transforms = self.build_transforms(hyp)

    def set_mixup(self, hyp):
        """For DETR sets the value of mixup to 0.5 and builds transformations."""
        hyp.mixup = 0.5
        self.transforms = self.build_transforms(hyp)

    def set_mosaic(self, hyp):
        """For DETR sets the probability of mosaic to 0.5 for selected epochs and builds transformations."""
        hyp.mosaic = 0.5
        self.transforms = self.build_transforms(hyp)

    def close_mixup(self, hyp):
        """For DETR sets mixup to 0.0 and builds transformations."""
        hyp.mixup = 0.0
        self.transforms = self.build_transforms(hyp)

    def change_scale(self, hyp):
        """Change scale to 0.9 after a certain number of epochs"""
        hyp.scale = 0.9
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances2d(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

class VajraSmallObjDetDataset(BaseDataset):
    def __init__(self, *args, data=None, task="small_obj_detect", **kwargs):
        self.use_keypoints = True
        self.use_segments = task == "segment"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],
                            bboxes=lb[:, 1:],
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING! No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for Vajra training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING! No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING! Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING! No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = vajra_transforms(self, self.img_size, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.img_size, self.img_size), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances2d(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

class MultiModalDataset(VajraDetDataset):
    def __init__(self, *args, data=None, task="detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)
    
    def update_labels_info(self, label):
        labels = super().update_labels_info(label)
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms
    
    @property
    def category_names(self):
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}
    
    @property
    def category_freq(self):
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq
    
    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(VajraDetDataset):
    def __init__(self, *args, task="detect", json_file = "", max_samples = 80, **kwargs):
        assert task in {"detect", "segment"}, "Currently only supports 'detect' and 'segment' tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        return []

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        x = {"labels": []}
        labels = []
        LOGGER.info("Loading annotations file...")
        with open(self.json_file, "r") as f:
            annotations = json.load(f)
        images = {f'{x["id"]:d}': x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            height, width, filename = img["height"], img["width"], img["filename"]
            im_file = Path(self.img_path) / f

            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []

            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(width)
                box[[1, 3]] /= float(height)

                if box[2] <= 0 or box[3] <= 0:
                    continue

                cat_name = " ".join([img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]])
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]
                box = [cls] + box.tolist()

                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([width, height], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([width, height], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            label = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]
                label = np.concatenate((classes.reshape(-1, 1), segments_to_boxes(segments)), 1)
            label = np.array(label, dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (height, width),
                    "cls": label[:, 0:1],  # n, 1
                    "bboxes": label[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x)
        return x
    
    def verify_labels(self, labels: list[dict[str, Any]]) -> None:
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")
    
    def get_labels(self) -> list[dict]:
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.json_file)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False
        [cache.pop(k) for k in ("hash", "version")]
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels
        
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq)
            )
            transforms.insert(-1, transform)
        return transforms
    
    @property
    def category_names(self):
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}
    
    @property
    def category_freq(self):
        categore_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    categore_freq[t] += 1
        return categore_freq
    
    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]

class VajraConcatDataset(ConcatDataset):
    @staticmethod
    def collate_fn(batch):
        return VajraDetDataset.collate_fn(batch)
    
    def close_mosaic(self, hyp: dict) -> None:
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    Extends torchvision ImageFolder to support Vajra classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize Vajra object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `img_size` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or args.cache == "ram"  # cache images into RAM
        self.cache_disk = args.cache == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.img_size,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.img_size, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        x["hash"] = get_hash([x[0] for x in self.samples])
        x["results"] = nf, nc, len(samples), samples
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return samples

class MultiLabelClassificationDataset(BaseDataset):
    def __init__(self, *args, data=None, augment=False, task="multilabel_classify", **kwargs):
        self.data = data
        self.use_keypoints = False
        self.task = task
        self.use_segments = False
        self.use_obb = False
        self.hyp = kwargs["hyp"]
        self.multi_label = True
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}

        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(0),
                    repeat(0),
                    repeat(self.multi_label),
                ),
            )

            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                    pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING! No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING! Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING! No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        scale = (1.0 - self.hyp.scale, 1.0)
        torch_transforms = (
            classify_augmentations(
                size=self.img_size,
                scale=scale,
                hflip=self.hyp.fliplr,
                vflip=self.hyp.flipud,
                erasing=self.hyp.erasing,
                auto_augment=self.hyp.auto_augment,
                hsv_h=self.hyp.hsv_h,
                hsv_s=self.hyp.hsv_s,
                hsv_v=self.hyp.hsv_v,
            )
            if self.augment
            else classify_transforms(size=self.img_size, crop_fraction=self.hyp.crop_fraction)
        )

        return torch_transforms

    def __getitem__(self, index):
        """Needed to override this method otherwise data is returned incorrectly."""
        current_data = self.get_image_and_label(index)
        sample_image = current_data["img"]
        sample_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
        transformed_image = self.transforms(sample_image)
        return {"img": transformed_image, "cls": current_data["cls"]}

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        self.hyp.mosaic = 0.0
        self.hyp.copy_paste = 0.0
        self.hyp.mixup = 0.0
        self.transforms = self.build_transforms(self.hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        label.pop("bboxes")  # np.array([[0, 0, 640, 640]])
        segments = label.pop("segments", [])
        label.pop("keypoints", None)
        label.pop("bbox_format")
        label.pop("normalized")
        current_classes = label.pop("cls")
        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = None
        # Need to convert the current classes to a vector of 0s and 1s for one-hot encoding
        class_vector = torch.zeros(self.data["nc"])
        for element in current_classes:
            class_vector[element] = 1
        label["cls"] = class_vector
        return label
        
# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    def __init__(self):
        super().__init__()
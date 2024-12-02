# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from pathlib import Path
from functools import partial

import torch

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

from vajra.utils import IterableSimpleNamespace, yaml_load
from vajra.checks import check_yaml

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}

def on_predict_start(predictor: object, persist: bool=False) -> None:
    if hasattr(predictor, "trackers") and persist:
        return 

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")
    
    trackers = []

    for _ in range(predictor.dataset.batch_size):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.batch_size

def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)

def register_tracker(model: object, persist: bool) -> None:
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_preprocess_end", partial(on_predict_postprocess_end, persist=persist))
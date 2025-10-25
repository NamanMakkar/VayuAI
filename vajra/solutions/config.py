from dataclasses import dataclass, field
from typing import Any
import cv2


@dataclass
class SolutionConfig:
    source: str | None = None
    model: str | None = None
    classes: list[int] | None = None
    show_conf: bool = True
    show_labels: bool = True
    region: list[tuple[int, int]] | None = None
    colormap: int | None = cv2.COLORMAP_DEEPGREEN
    show_in: bool = True
    show_out: bool = True
    up_angle: float = 145.0
    down_angle: int = 90
    kpts: list[int] = field(default_factory=lambda: [6, 8, 10])
    analytics_type: str = "line"
    figsize: tuple[int, int] | None = (12.8, 7.2)
    blur_ratio: float = 0.5
    vision_points: tuple[int, int] = (20, 20)
    crop_dir: str = "cropped_detections"
    json_file: str = None
    line_width: int = 2
    records: int = 5
    fps: float = 30.0
    max_hist: int = 5
    meter_per_pixel: float = 0.05
    max_speed: int = 120
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: str | None = None
    max_det: int = 300
    half: bool = False
    tracker: str = "botsort.yaml"
    verbose: bool = True
    data: str = "images"

    def update(self, **kwargs: Any):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"{key} is not a valid solution argument")
        return self
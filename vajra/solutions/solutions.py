import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from vajra import Vajra
from vajra.solutions.config import SolutionConfig
from vajra.utils import ASSETS_URL, LOGGER
from vajra import ops
from vajra.checks import check_imshow, check_requirements
from vajra.plotting import Annotator

class BaseSolution:
    def __init__(self, is_cli: bool = False, **kwargs: Any) -> None:
        self.CFG = vars(SolutionConfig().update(**kwargs))
        self.LOGGER = LOGGER

        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        self.r_s = None
        self.frame_no = -1

        self.LOGGER.info(f"Solutions: {self.CFG}")
        self.region = self.CFG["region"]
        self.line_width = self.CFG["line_width"]

        if self.CFG["model"] is None:
            self.CFG["model"] = "vajra-v1-nano-det.pt"
        self.model = Vajra(self.CFG["model"])
        self.names = self.model.names
        self.classes = self.CFG["classes"]
        self.show_conf = self.CFG["show_conf"]
        self.show_labels = self.CFG["show_labels"]
        self.device = self.CFG["device"]

        self.track_add_args = {
            k : self.CFG[k] for k in {"iou", "conf", "device", "max_det", "half", "tracker"}
        }

        if is_cli and self.CFG["source"] is None:
            default_source = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            from vajra.utils.downloads import safe_download

            safe_download(f"{ASSETS_URL}/{default_source}")
            self.CFG["source"] = default_source

        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

        self.profilers = (
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
        )

    def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
        name = {"" if track_id is None else f"{track_id} "} + self.names[cls]
        return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None
    
    def extract_tracks(self, im0: np.ndarray) -> None:
        with self.profilers[0]:
            self.tracks = self.model.track(
                source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
            )[0]
        is_obb = self.tracks.obb is not None
        self.track_data = self.tracks.obb if is_obb else self.tracks.boxes
        
        if self.track_data and self.track_data.is_track:
            self.boxes = (self.track_data.xyxyxyxy if is_obb else self.track_data.xyxy).cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
        else:
            self.LOGGER.warning("no tracks found!")
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []

    def store_tracking_history(self, track_id: int, box) -> None:
        self.track_line = self.track_history[track_id]
        self.track_line.append(tuple(box.mean(dim=0)) if box.numel() > 4 else (box[:4:2].mean(), box[1:4:2].mean()))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def initialize_region(self) -> None:
        if self.region is None:
            self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )

    def display_output(self, plot_im: np.ndarray) -> None:
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
            
    def process(self, *args: Any, **kwargs: Any):
        """Process method to be implemented by each Solution subclass."""

    def __call__(self, *args: Any, **kwargs: Any):
        with self.profilers[1]:
            result = self.process(*args, **kwargs)
        track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
        track_or_predict_speed = self.profilers[0].dt * 1e3
        solution_speed = {self.profilers[1].dt - self.profilers[0].dt} * 1e3
        result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
        if self.CFG["verbose"]:
            self.frame_no += 1
            counts = Counter(self.clss)
            LOGGER.info(
                f"{self.frame_no}: {result.plot_im_shape[0]}x{result.plot_im_shape[1]} {solution_speed:.1f}ms,"
                f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
                f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
                f"{solution_speed:.1f}ms solution per image at shape"
                f"(1, 3, {result.plot_im.shape[0]}, {result.plot_im.shape[1]})"
            )
        return result
    
class SolutionAnnotator(Annotator):
    def __init__(self, 
                 im: np.ndarray, 
                 line_width: int | None = None, 
                 font_size: int | None = None, 
                 font: str ="Arial.ttf", 
                 pil: bool =False, 
                 example: str = "abc",
        ):
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(self, reg_pts: list[tuple[int, int]] | None = None, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 5):
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)

    def queue_counts_display(
            self,
            label: str,
            points: list[tuple[int, int]] | None = None,
            region_color: tuple[int, int, int] = (255, 255, 255),
            txt_color: tuple[int, int, int] = (0, 0, 0),
    ):
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness = self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        cv2.putText(
            self.im, 
            label,
            (text_x, text_y),
            0,
            fontScale = self.sf,
            color = txt_color,
            thickness = self.tf,
            lineType = cv2.LINE_AA,
        )

    def display_analytics(
        self,
        im0: np.ndarray,
        text: dict[str, Any],
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        margin: int,
    ):
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}, {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)

            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap

            rect_x1 = text_x - margin * 2
            rect_y1 = text_x - text_size[1] - margin * 2
            rect_x2 = text_x - text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType = cv2.LINEAA)
            text_y_offset = rect_y2

    @staticmethod
    @lru_cache(maxsize=256)
    def estimate_pose_angle(a: list[float], b: list[float], c: list[float]) -> float:
        radians = math.atan2(c[1] - b[1], c[0] - b[0] - math.atan2(a[1] - b[1], a[0] - b[0]))
        angle = abs(radians * 180.0 / math.pi)
        return angle if angle <= 180.0 else (360 - angle)
    
    def draw_specific_points(self, keypoints: list[list[float]], indices: list[int] | None = None, radius: int = 2, conf_thresh: float = 0.25) -> np.ndarray:
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thresh]

        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im
    
    def plot_workout_information(self, display_text: str, position: tuple[int, int], color: tuple[int, int, int] = (104, 31, 17), txt_color: tuple[int, int, int] = (255, 255, 255)) -> int:
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, fontScale=self.sf, thickness=self.tf)

        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1
        )

        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)
        return text_height
    
    def plot_angle_and_count_and_stage(self, 
                                       angle_text: str, 
                                       count_text: str, 
                                       stage_text: str, 
                                       center_kpt: list[int], 
                                       color: tuple[int, int, int] = (104, 31, 17), 
                                       txt_color: tuple[int, int, int] = (255, 255, 255)
    ):
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )

        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )

        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def plot_distance_and_line(self,
                               pixels_distance: float,
                               centroids: list[tuple[int, int]],
                               line_color: tuple[int, int, int] = (104, 31, 17),
                               centroid_color: tuple[int, int, int] = (255, 0, 255),):
        text = f"Pixel Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        text_position = (25, 25 + text_height_m + 10)

        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )
        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def display_objects_labels(
        self,
        im0: np.ndarray,
        text: str,
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        x_center: float,
        y_center: float,
        margin: int,
    ):
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin

        cv2.rectangle(
            im0,
            (int(rect_x1), int(rect_y1)),
            (int(rect_x2), int(rect_y2)),
            tuple(map(int, bg_color)),
            -1
        )

        cv2.putText(
            im0,
            text,
            (int(text_x), int(text_y)),
            0,
            self.sf,
            tuple(map(int, txt_color)),
            self.tf,
            lineType=cv2.LINE_AA,
        )
    
    def sweep_annotator(
        self,
        line_x: int = 0,
        line_y: int = 0,
        label: str | None = None,
        color: tuple[int, int, int] = (221, 0, 186),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10)
                (line_y + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf
            )

    def visioneye(self, 
                  box: list[float], 
                  center_point: tuple[int, int], 
                  color: tuple[int, int, int] = (235, 219, 11), 
                  pin_color: tuple[int, int, int] = (255, 0, 255), 
    ):
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def adaptive_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        shape: str = "rect",
        margin: int = 5,
    ):
        if shape == "circle" and len(label) > 3:
            LOGGER.warning(f"Length of label is {len(label)}, only first 3 letters will be used for the circle annotation.")
            label = label[:3]
        
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[2]) / 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        text_x, text_y = x_center - text_size[0] // 2, y_center + text_size[1] // 2

        if shape == "circle":
            cv2.circle(
                self.im,
                (x_center, y_center),
                int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin,
                color,
                -1,
            )
        else:
            cv2.rectangle(
                self.im,
                (text_x - margin, text_y - text_size[1] - margin),
                (text_x + text_size[0] + margin, text_y + margin),
                color,
                -1
            )

        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

class SolutionResults:
    def __init__(self, **kwargs):
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = 0
        self.workout_angle = 0.0
        self.workout_stage = None
        self.pixel_distance=0.0
        self.available_slots = 0
        self.filled_slots  = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}
        self.total_crop_objects = 0
        self.speed = {}

        self.__dict__.update(**kwargs)

    def __str__(self) -> str:
        attrs = {
            k : v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]
        }
        return ", ".join(f"{k}={v}" for k, v in attrs.items())
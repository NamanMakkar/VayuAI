import math
from typing import Any

import cv2

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class DistanceCalculation(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.left_mouse_count = 0
        self.selected_boxes: dict[int, list[float]] = {}
        self.centroids: list[list[int]] = []

    def mouse_event_for_distance(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def process(self, im0) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        pixels_distance = 0

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(box, color=colors(int(cls), True), label=self.adjust_box_label(cls, conf, track_id))

            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box
        
        if len(self.selected_boxes) == 2:
            self.centroids.extend(
                [[int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)] for box in self.selected_boxes.values()]
            )

            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            annotator.plot_distance_and_line(pixels_distance, self.centroids)
        
        self.centroids = []
        plot_im = annotator.result()
        self.display_output(plot_im)
        if self.CFG.get("show") and self.env_check:
            cv2.setMouseCallback("VayuAI Solutions", self.mouse_event_for_distance)

        return SolutionResults(plot_im=plot_im, pixels_distance=pixels_distance, total_tracks=len(self.track_ids))
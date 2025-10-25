from collections import deque
from math import sqrt
from typing import Any

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.fps = self.CFG["fps"]
        self.frame_count = 0
        self.trk_frame_ids = {}
        self.speed_per_obj = {}
        self.trk_hist = {}
        self.locked_ids = set()
        self.max_hist = self.CFG["max_hist"]
        self.meter_per_pixel = self.CFG["meter_per_pixel"]
        self.max_speed = self.CFG["max_speed"]

    def process(self, im0) -> SolutionResults:
        self.frame_count += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, track_id, _, _ in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_hist:
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)
                self.trk_frame_ids[track_id] = self.frame_count

            if track_id not in self.locked_ids:
                trk_hist = self.trk_hist[track_id]
                trk_hist.append(self.track_line[-1])

                if len(trk_hist) == self.max_hist:
                    p0, p1 = trk_hist[0], trk_hist[1]
                    dt = (self.frame_count - self.trk_frame_ids[track_id]) / self.fps

                    if dt > 0:
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                        pixel_distance = sqrt(dx * dx + dy * dy)
                        meters = pixel_distance * self.meter_per_pixel
                        self.speed_per_obj[track_id] = int(
                            min((meters / dt) * 3.6, self.max_speed)
                        )
                        self.locked_ids.add(track_id)
                        self.trk_hist.pop(track_id, None)
                        self.trk_frame_ids.pop(track_id, None)
            
            if track_id in self.speed_per_obj:
                speed_label = f"{self.speed_per_obj[track_id]} km/h"
                annotator.box_label(box, label=speed_label, color=colors(track_id, True))
        
        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
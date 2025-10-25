from typing import Any

import cv2
import numpy as np

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class TrackZone(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        default_region = [(75, 75), (565, 285), (75, 285)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))
        self.mask = None

    def process(self, im0: np.ndarray) -> SolutionResults:
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        if self.mask is None:
            self.mask = np.zeros_like(im0[:, :, 0])
            cv2.fillPoly(self.mask, [self.region], 255)
        masked_frame = cv2.bitwise_and(im0, im0, mask=self.mask)
        self.extract_tracks(masked_frame)

        cv2.polylines(im0, [self.region], isClose=True, color=(255, 255, 255), thickness=self.line_width)

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf, track_id=track_id), color=colors(track_id, True)
            )
        
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
from typing import Any
import numpy as np
from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class VisionEye(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision_point = self.CFG["vision_point"]

    def process(self, im0: np.ndarray) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(int(track_id), True))
            annotator.visioneye(box, self.vision_point)

        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(plot_im=plot_im, total_track=len(self.track_ids))

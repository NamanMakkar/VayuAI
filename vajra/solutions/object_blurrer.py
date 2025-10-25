from typing import Any

import cv2

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.utils import LOGGER
from vajra.plotting import colors

class ObjectBlurrer(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        blur_ratio = self.CFG["blur_ratio"]

        if blur_ratio < 0.1:
            LOGGER.warning("blur ratio cannot be less than 0.1, updating it to default value 0.5")
            blur_ratio = 0.5
        self.blur_ratio = int(blur_ratio * 100)

    def process(self, im0) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)
        for box, cls, conf in zip(self.boxes, self.clss, self.confs):
            blur_obj = cv2.blur(
                im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])],
                (self.blur_ratio, self.blur_ratio),
            )

            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            annotator.box_label(box, label=self.adjust_box_label(cls, conf), color=colors(cls, True))

        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
from typing import Any

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class QueueManager(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0
        self.rect_color = (255, 255, 255)
        self.region_length = len(self.region)

    def process(self, im0) -> SolutionResults:
        self.counts = 0
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        annotator.draw_region(reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2)

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            self.store_tracking_history(track_id, box)

            track_history = self.track_history.get(track_id, [])

            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

            annotator.queue_counts_display(
                f"Queue Counts: {str(self.counts)}",
                points=self.region,
                region_color=self.rect_color,
                txt_color=(104, 31, 17),
            )

            plot_im = annotator.result()
            self.display_output(plot_im)

            return SolutionResults(plot_im=plot_im, queue_count=self.counts, total_tracks=len(self.track_ids))
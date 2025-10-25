from typing import Any
import cv2

import numpy as np
from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.solutions.object_counter import ObjectCounter

class Heatmap(ObjectCounter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initialized = False
        if self.region is not None:
            self.initialize_region()

        self.colormap = self.CFG["colormap"]
        self.heatmap = None
    
    def heatmap_effect(self, box: list[float]) -> None:
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # Calculate squared distances from the center
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # Create a mask of points within the radius
        within_radius = dist_squared <= radius_squared

        # Update only the values within the bounding box in a single vectorized operation
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def process(self, im0: np.ndarray) -> SolutionResults:
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99
            self.initialized = True

        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, track_id, cls, in zip(self.boxes, self.track_ids, self.clss):
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)

                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)
        
        plot_im = self.annotator.result()
        if self.region is not None:
            self.display_counts(plot_im)

        if self.track_data.is_track:
            normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.colormap)
            plot_im = cv2.addWeighted(plot_im, 0.5, colored_heatmap, 0.5, 0)
        
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
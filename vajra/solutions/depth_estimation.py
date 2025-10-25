import torch

import cv2
import numpy as np
from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from transformers import pipeline

class DetectionDepthSolution(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_model = pipeline("depth-estimation", model="apple/ml-depth-pro", device=0 if torch.cuda.is_available() else -1)
        self.LOGGER.info("Apple Depth Pro model initialized")

    def process(self, im0):
        self.extract_tracks(im0)

        depth_result = self.depth_model(im0)
        depth_map = np.array(depth_result["depth"], dtype=np.float32)

        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        annotator = SolutionAnnotator(im0.copy(), line_width=self.line_width)

        for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
            x1, y1, x2, y2 = map(int, box)
            label = self.adjust_box_label(cls, conf=self.confs[self.track_ids.index(track_id)], track_id=track_id)
            object_depth = depth_map[y1:y2, x1:x2]
            avg_depth = np.median(object_depth) if object_depth.size > 0 else 0
            label = f"{label} {avg_depth: .2f}m"
            annotator.box_label([x1, y1, x2, y2], label, color=(0, 255, 0))
        
        combined = np.hstack((annotator.im, depth_vis_color))
        self.display_output(combined)
        return SolutionResults(plot_im = combined)
        
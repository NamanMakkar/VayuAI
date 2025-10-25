from typing import Any

import numpy as np

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class RegionCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()

    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region
    
    def initialize_regions(self):
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):
            self.region = {"Region#01": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            region = self.add_region(name, pts, colors(i, True), (255, 255, 255))
            region["prepared_polygon"] = self.prep(region["polygon"])

    def process(self,im0: np.ndarray) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            center = self.Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            annotator.adaptive_label(
                [x1, y1, x2, y2],
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
                shape="rect",
            )
        
        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)

    
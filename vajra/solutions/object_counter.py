from collections import defaultdict

from typing import Any
from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.plotting import colors

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self.region_initialized = False

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2

    def count_objects(
        self,
        current_centroid: tuple[float, float],
        track_id: int,
        prev_position: tuple[float, float] | None,
        cls: int,
    ) -> None:
        if prev_position is None or track_id in self.counted_ids:
            return
        
        if len(self.region) == 2: #Linear region (defined as a line segment)
            if self.r_s.intersects(self.LineString([prev_position, current_centroid])):
                # Determine orientation of the region (vertical or horizontal)
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x coordinates to determine direction
                    if current_centroid[0] > prev_position[0]:
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                # Horizontal region: Compare y coordinates to determine direction
                elif current_centroid[1] > prev_position[1]:
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1

                self.counted_ids.append(track_id)
        
        elif len(self.region) > 2:
            if self.r_s.contains(self.Point(current_centroid)):
                # Determine motion direction for vertical or horizontal polygons
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (region_width < region_height 
                    and current_centroid[0] > prev_position[0]
                    or region_width >= region_height
                    and current_centroid[1] > prev_position[1]
                ):
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1

                else:
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def display_counts(self, plot_im) -> None:
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if value["IN"] != 0 or value["OUT"] != 0 and (self.show_in or self.show_out)
        }

        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0) -> SolutionResults:
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True
        
        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=self.classwise_count,
            total_tracks=len(self.track_ids),
        )
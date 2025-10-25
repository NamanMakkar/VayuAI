from collections import defaultdict

from typing import Any
from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults

class AIGym(BaseSolution):
    def __init__(self, **kwargs):
        kwargs["model"] = kwargs.get("model", "vajra-v1-nano-pose.pt")
        super().__init__(**kwargs)
        self.states = defaultdict(lambda: {"angle": 0, "count": 0, "stage": "-"})

        self.up_angle = float(self.CFG["up_angle"])
        self.down_angle = float(self.CFG["down_angle"])
        self.kpts = self.CFG["kpts"]

    def process(self, im0) -> SolutionResults:
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        if len(self.boxes):
            kpt_data = self.tracks.keypoints.data

            for i, k in enumerate(kpt_data):
                state = self.states[self.track_ids[i]]
                state["angle"] = annotator.estimate_pose_angle(*[k[int(idx)] for idx in self.kpts])
                annotator.draw_specific_points(k, self.kpts, radius=self.line_width * 3)

                if state["angle"] < self.down_angle:
                    if state["stage"] == "up":
                        state["count"] += 1
                    state["stage"] = "down"
                elif state["angle"] > self.up_angle:
                    state["stage"] = "up"

                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=state["angle"],
                        count_text=state["count"],
                        stage_text=state["stage"],
                        center_kpt=k[int(self.kpts[1])]
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            workout_count=[v["count"] for v in self.states.values()],
            workout_stage=[v["stage"] for v in self.states.values()],
            workout_angle=[v["angle"] for v in self.states.values],
            total_tracks=len(self.track_ids),
        )
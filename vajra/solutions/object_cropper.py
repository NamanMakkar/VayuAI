import os
from pathlib import Path
from types import Any

from vajra.solutions.solutions import BaseSolution, SolutionResults
from vajra.plotting import save_one_box

class ObjectCropper(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crop_dir = self.CFG["crop_dir"]
        if not os.path.exists(self.crop_dir):
            os.mkdir(self.crop_dir)
        if self.CFG["show"]:
            self.LOGGER.warning(
                f"show=True disabled for crop solution, results will be saved in the directory named: {self.crop_dir}"
            )

        self.crop_idx = 0
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"]

    def process(self, im0) -> SolutionResults:
        with self.profilers[0]:
            results = self.model.predict(
                im0,
                classes=self.classes,
                conf = self.conf,
                iou=self.iou,
                device=self.CFG["device"],
                verbose=False
            )[0]
            self.clss=results.boxes.cls.tolist()

        for box in results.boxes:
            self.crop_idx += 1
            save_one_box(
                box.xyxy,
                im0,
                file=Path(self.crop_dir) / f"crop_{self.crop_idx}.jpg",
                BGR=True,
            )
        return SolutionResults(plot_im = im0, total_crop_objects = self.crop_idx)
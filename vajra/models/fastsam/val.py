# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

from vajra.models.vajra.segment import SegmentationValidator
from vajra.metrics import SegmentationMetrics

class FastSAMValidator(SegmentationValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False
        self.metrics = SegmentationMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
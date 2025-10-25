from copy import copy, deepcopy

from vajra.models.vajra.segment import SegmentationTrainer
from vajra.nn.vajra import VajraESegmentationModel
from vajra.utils import RANK

from .train import VajraETrainer, VajraETrainerFromScratch, VajraEVPTrainer
from .val import VajraESegmentationValidator

class VajraESegmentationTrainer(VajraETrainer, SegmentationTrainer):
    def get_model(self, model_name=None, weights=None, verbose=True):
        model = VajraESegmentationModel(
            model_name,
            num_classes=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model
    
    def get_validator(self):
        self.loss_names = "box", "seg", "cls", "dfl"
        return VajraESegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
class VajraEPESegmentationTrainer(SegmentationTrainer):
    def get_model(self, model_name=None, weights=None, verbose=True):
        model = VajraESegmentationModel(
            model_name,
            num_classes=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)
        
        model.eval()
        names = list(self.data["names"].values())
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].branch_cls[0][2] = deepcopy(model.model[-1].branch_cls[0][2]).requires_grad_(True)
        model.model[-1].branch_cls[1][2] = deepcopy(model.model[-1].branch_cls[1][2]).requires_grad_(True)
        model.model[-1].branch_cls[2][2] = deepcopy(model.model[-1].branch_cls[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model
    
class VajraESegmentationTrainerFromScratch(VajraETrainerFromScratch, VajraESegmentationTrainer):
    pass

class VajraESegmentationVPTrainer(VajraEVPTrainer, VajraESegmentationTrainerFromScratch):
    pass

# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License
__version__ = "1.0.4"

from vajra.models import Vajra, VajraE, VajraWorldModel, SAM, FastSAM, DETR
from vajra.utils import ASSETS, SETTINGS as settings
#from vajra.checks import check_vajra as checks
from vajra.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "Vajra",
    "VajraE",
    "VajraWorldModel",
    "SAM",
    "FastSAM",
    "DETR"
)
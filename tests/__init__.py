from vajra.utils import ASSETS, ROOT, WEIGHTS_DIR
from vajra import checks

MODEL = WEIGHTS_DIR / "" / "vajra-v1-nano-det.pt"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
TMP = (ROOT / "../tests/tmp").resolve()
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()

__all__ = (
    "MODEL",
    "SOURCE",
    "SOURCES_LIST",
    "TMP",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)
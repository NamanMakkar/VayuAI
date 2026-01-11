

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from vajra import checks
from vajra.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(self, texts):
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        pass

class CLIP(TextModel):
    def __init__(self, size: str, device: torch.device) -> None:
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str]) -> torch.Tensor:
        return clip.tokenize(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats
    
    @smart_inference_mode()
    def encode_images(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats

class MobileCLIP(TextModel):
    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import mobileclip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip
        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from vajra import download
            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features
    
class MobileCLIPTS(TextModel):
    def __init__(self, device: torch.device):
        super().__init__()
        from vajra.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset("mobileclip_blt.ts"), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device=device
    
    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.encoder(texts).to(dtype)
    
def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    else:
        raise ValueError(f"Unrecognized base model: '{base}'. Supported base models: 'clip', 'mobileclip'.")
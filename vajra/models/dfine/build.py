# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path
from vajra.nn.transformer import HybridEncoder
from vajra.models.utils import DEIMCriterion, DFINECriterion
from vajra.nn.head import DFINETransformer
from vajra.nn.backbones.hgnet.hgnetv2 import HGNetV2
from vajra.utils.dfine_ops import HungarianMatcher
from vajra.utils.torch_utils import model_info, initialize_weights, fuse_conv_and_bn, time_sync, intersect_dicts, scale_img
from vajra.utils import LOGGER, HYPERPARAMS_DETR_CFG_DICT, HYPERPARAMS_DETR_CFG, ROOT
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.vajra import get_task
from vajra.nn.modules import ConvBNAct, DepthwiseConvBNAct, RepVGGDW, RepVGGBlock

try:
    import thop
except ImportError:
    thop = None

def build_dfine_deim_n(checkpoint=None, model_name="dfine-nano-det", num_classes=80):
    backbone_config = {
        "name": "B0",
        "return_idx": [2, 3],
        "freeze_at": -1,
        "freeze_norm": False,
        "use_lab": True,
        "pretrained": True if checkpoint is not None else False,
        "local_model_dir": ROOT
    }

    hybrid_encoder_config = {
        "in_channels": [512, 1024],
        "feat_strides": [16, 32],
        "hidden_dim": 128,
        "use_encoder_idx": [1],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.,
        "enc_act": nn.GELU(),
        "expansion": 0.34,
        "depth_mul": 0.5,
        "act": nn.SiLU(),
    }

    transformer_head_config = {
        "num_classes": num_classes,
        "feat_channels": [128, 128],
        "feat_strides": [16, 32],
        "hidden_dim": 128,
        "dim_feedforward": 512,
        "num_levels": 2,
        "num_layers": 3,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [6, 6],
        "cross_attn_method": "default",
        "query_select_method": "default",
    }
    return _build_dfine(backbone_config, hybrid_encoder_config, transformer_head_config, checkpoint=checkpoint, model_name=model_name)

def build_dfine_deim_s(checkpoint=None, model_name="dfine-small-det", num_classes=80):
    backbone_config = {
        "name": "B0",
        "return_idx": [1, 2, 3],
        "freeze_at": -1,
        "freeze_norm": False,
        "use_lab": True,
        "pretrained": True if checkpoint is not None else False,
        "local_model_dir": ROOT
    }

    hybrid_encoder_config = {
        "in_channels": [256, 512, 1024],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.,
        "enc_act": nn.GELU(),
        "depth_mul": 0.34,
        "expansion": 0.5,
        "act": nn.SiLU(),
    }

    transformer_head_config = {
        "num_classes": num_classes,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "num_layers": 3,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
    }
    return _build_dfine(backbone_config, hybrid_encoder_config, transformer_head_config, checkpoint, model_name)

def build_dfine_deim_m(checkpoint=None, model_name="dfine-medium-det", num_classes=80):
    backbone_config = {
        "name": "B2",
        "return_idx": [1, 2, 3],
        "freeze_at": -1,
        "freeze_norm": False,
        "use_lab": True,
        "pretrained": True if checkpoint is not None else False,
        "local_model_dir": ROOT
    }

    hybrid_encoder_config = {
        "in_channels": [384, 768, 1536],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.,
        "enc_act": nn.GELU(),
        "depth_mul": 0.67,
        "expansion": 1.0,
        "act": nn.SiLU(),
    }

    transformer_head_config = {
        "num_classes": num_classes,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "num_layers": 4,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",

    }
    return _build_dfine(backbone_config, hybrid_encoder_config, transformer_head_config, checkpoint, model_name)

def build_dfine_deim_l(checkpoint=None, model_name="dfine-large-det", num_classes=80):
    backbone_config = {
        "name": "B4",
        "return_idx": [1, 2, 3],
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": True,
        "use_lab": True,
        "pretrained": True if checkpoint is not None else False,
        "local_model_dir": ROOT,
    }

    hybrid_encoder_config = {
        "in_channels": [512, 1024, 2048],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.,
        "enc_act": nn.GELU(),
        "expansion": 1.0,
        "depth_mul": 1,
        "act": nn.SiLU(),
    }

    transformer_head_config = {
        "num_classes": num_classes,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "num_layers": 6,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
    }
    return _build_dfine(backbone_config, hybrid_encoder_config, transformer_head_config, checkpoint, model_name)

def build_dfine_deim_xl(checkpoint=None, model_name="dfine-xlarge-det", num_classes=80):
    backbone_config = {
        "name": "B5",
        "return_idx": [1, 2, 3],
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": True,
        "use_lab": True,
        "pretrained": True if checkpoint is not None else False,
        "local_model_dir": ROOT,
    }

    hybrid_encoder_config = {
        "in_channels": [512, 1024, 2048],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 384,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.,
        "enc_act": nn.GELU(),
        "expansion": 1.0,
        "depth_mul": 1,
        "act": nn.SiLU(),
    }

    transformer_head_config = {
        "num_classes": num_classes,
        "feat_channels": [384, 384, 384],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "num_layers": 6,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 8,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
    }
    return _build_dfine(backbone_config, hybrid_encoder_config, transformer_head_config, checkpoint=checkpoint, model_name=model_name)

detr_model_map = {
    "dfine-nano-det": build_dfine_deim_n,
    "dfine-small-det": build_dfine_deim_s,
    "dfine-medium-det": build_dfine_deim_m,
    "dfine-large-det": build_dfine_deim_l,
    "dfine-xlarge-det": build_dfine_deim_xl,
    "deim-nano-det": build_dfine_deim_n,
    "deim-small-det": build_dfine_deim_s,
    "deim-medium-det": build_dfine_deim_m,
    "deim-large-det": build_dfine_deim_l,
    "deim-xlarge-det": build_dfine_deim_xl,
}

def _build_dfine(backbone_config: Dict, hybrid_encoder_config: Dict, transformer_head_config: Dict, checkpoint=None, model_name="dfine-nano-det"):
    args = HYPERPARAMS_DETR_CFG #{**HYPERPARAMS_DETR_CFG_DICT}
    #LOGGER.info(f"Print args keys: {args.keys()}\n")
    backbone = HGNetV2(**backbone_config)
    encoder = HybridEncoder(in_channels=hybrid_encoder_config["in_channels"],
                            feat_strides=hybrid_encoder_config["feat_strides"],
                            hidden_dim=hybrid_encoder_config["hidden_dim"],
                            nhead=hybrid_encoder_config["nhead"],
                            dim_feedforward=hybrid_encoder_config["dim_feedforward"],
                            dropout=hybrid_encoder_config["dropout"],
                            encoder_act=hybrid_encoder_config["enc_act"],
                            use_encoder_idx=hybrid_encoder_config["use_encoder_idx"],
                            num_encoder_layers=hybrid_encoder_config["num_encoder_layers"],
                            expansion=hybrid_encoder_config["expansion"],
                            depth_mul=hybrid_encoder_config["depth_mul"],
                            act=hybrid_encoder_config["act"],
                            eval_spatial_size=[args.img_size, args.img_size])

    decoder = DFINETransformer(num_classes=transformer_head_config["num_classes"], 
                               num_queries=transformer_head_config["num_queries"],
                               feat_channels=transformer_head_config["feat_channels"],
                               feat_strides=transformer_head_config["feat_strides"],
                               hidden_dim=transformer_head_config["hidden_dim"],
                               num_levels=transformer_head_config["num_levels"],
                               num_points=transformer_head_config["num_points"],
                               num_layers=transformer_head_config["num_layers"],
                               num_denoising=transformer_head_config["num_denoising"], 
                               label_noise_ratio=transformer_head_config["label_noise_ratio"], 
                               box_noise_scale=transformer_head_config["box_noise_scale"],
                               eval_idx=transformer_head_config["eval_idx"],
                               eval_spatial_size=[args.img_size, args.img_size],
                               cross_attn_method=transformer_head_config["cross_attn_method"],
                               query_select_method=transformer_head_config["query_select_method"],
                               reg_max=transformer_head_config["reg_max"],
                               reg_scale=transformer_head_config["reg_scale"],
                               layer_scale=transformer_head_config["layer_scale"],
                               activation=nn.SiLU() if "deim" in model_name else nn.ReLU(),
                               mlp_act="silu" if "deim" in model_name else "relu",)
    
    model_name_type = model_name.split("-")[0]
    if model_name_type == "deim":
        model = DEIM(backbone, encoder, decoder)
    elif model_name_type == "dfine":
        model = DFINE(backbone, encoder, decoder)
    
    if checkpoint is not None:
        checkpoint = attempt_download_asset(checkpoint)
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    return model

def build_rtdetr(task="detect", num_classes=80, size="nano", verbose=False, model_name="dfine-nano-det"):
    model_name_type = model_name.split("-")[0]
    model_map_key = str(Path(model_name).stem) if Path(model_name).suffix else model_name
    build_func = detr_model_map[model_map_key]
    checkpoint = model_name if Path(model_name).suffix else None #str(Path(model_name).with_suffix(".pt"))
    model = build_func(checkpoint, model_name=model_map_key, num_classes=num_classes)
    np_backbone = sum(x.numel() for x in model.backbone.parameters())
    np_encoder = sum(x.numel() for x in model.encoder.parameters())
    np_head = sum(x.numel() for x in model.decoder.parameters())
    nps = [np_backbone, np_encoder, np_head]
    if verbose:
        LOGGER.info(f"Task: {task}; Number of Classes: {num_classes}\n\n\n")
        LOGGER.info(f"Building DETR; type: {model_name_type.upper()} ...\n\n")
        LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<25}{'arguments':<30}\n")
        idx_counter = 0
        for i, (_, module) in enumerate(model.named_children()):
            module_info, args_info = module.get_module_info()
            LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{nps[i]:>10}  {module_info:<25}{args_info:<30}")
            idx_counter += 1
        
        LOGGER.info(f"\nBackbone Parameters: {np_backbone}\n\n")
        LOGGER.info(f"Hybrid Encoder Parameters: {np_encoder}\n\n")
        LOGGER.info(f"Transformer Head Parameters: {np_head}\n\n")
        LOGGER.info(f"{model_name_type.upper()}-{size}; Task: {task}; Total Parameters: {np_backbone + np_encoder + np_head}\n\n")
    #model = nn.Sequential(model)

    losses = ["mal", "boxes", "local"] if model_name_type == "deim" else ["vfl", "boxes", "local"]
    loss_weight_dict = {"loss_mal": 1., "loss_bbox": 5., "loss_giou": 2., "loss_fgl": 0.15, "loss_ddf": 1.5,} if model_name_type == "deim" else {"loss_vfl": 1., "loss_bbox": 5., "loss_giou": 2., "loss_fgl": 0.15, "loss_ddf": 1.5}
    matcher_weight_dict = {"class": 2., "bbox": 5., "giou": 2.}
    matcher_config = {
        "matcher_weight_dict": matcher_weight_dict,
        "alpha": 0.25,
        "gamma": 2.0
    }
    loss_config = {
        "losses": losses,
        "use_focal_loss": True,
        "loss_weight_dict": loss_weight_dict,
        "alpha": 0.75,
        "gamma": 2.0,
    }
    return model, loss_config, matcher_config

class DEIM(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder", ]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder:nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder", ]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x
    
class RTDETR_Model(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        if augment:
            return self._predict_augment(x, batch)

        return self._predict_once(x, profile, visualize, batch)
    
    def _predict_augment(self, x, batch=None):
        LOGGER.warning(
            f"WARNING! Augmented Inference is task specific"
            f"Switching to single-scale inference"
        )
        return self._predict_once(x, batch)

    def _predict_once(self, x, profile=False, visualize=False, batch=None, embed=None):
        dt = []
        y = []
        for n, m in self.model.named_children():
            if profile:
                self._profile_one_layer(m, x, dt)
            if n != "decoder":
                x = m(x)
                y.append(x)
            else:
                x = m(y[-1], targets=batch)

        #head = self.model[0].decoder
        #x = head(y[-1], targets=batch)
        return x
    
    def _profile_one_layer(self, layer, x, dt):
        num_params_layer = sum(x.numel() for x in layer.parameters())
        copy_input = layer == self.model.decoder and isinstance(x, list)
        flops = thop.profile(layer, inputs=[x.copy() if copy_input else x], verbose=False)[0]
        t = time_sync()
        for _ in range(10):
            layer(x.copy() if copy_input else x)
        dt.append((time_sync() - t) * 100)
        if layer == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {num_params_layer:10.0f}  {str(layer)}')
        if copy_input:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (ConvBNAct, DepthwiseConvBNAct)) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGBlock):
                    m.convert_to_deploy()
            self.info(verbose=verbose)
        return self
    
    def is_fused(self, threshold=10):
        batchnorms = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, batchnorms) for v in self.modules()) < threshold
    
    def init_criterion(self):
        return NotImplementedError, "Criterion is task specific"
    
    def loss(self, batch, preds=None):
        return NotImplementedError, "Loss will be model specific"
    
    def info(self, detailed = False, verbose=True, img_size=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=img_size)
    
class RTDETR_DetModel(RTDETR_Model):
    def __init__(self, model_name="dfine-nano-det", num_classes=None, verbose=True):
        super().__init__()
        self.model_name = model_name
        self.model_name_type = model_name.split("-")[0]
        self.num_classes = 80
        size = model_name.split("-")[-2]
        task = get_task(model_name)
        if num_classes and num_classes != self.num_classes:
            LOGGER.info(f"Overriding num_classes={self.num_classes} with num_classes={num_classes}")
            self.num_classes = num_classes

        self.model, self.loss_config, self.matcher_config = build_rtdetr(task, self.num_classes, size, verbose, model_name)
        head = self.model.decoder
        self.num_classes = num_classes
        self.reg_max = head.reg_max
        self.stride = torch.Tensor([32])
        if verbose:
            self.info()
            LOGGER.info('')

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "labels": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "boxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups
        }
        preds = self.predict(img, batch=targets) if preds is None else preds
        out = preds if self.training else preds[1]

        loss = self.criterion(out, targets)

        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in self.loss_config["losses"]], device=img.device,
        )
    
    def init_criterion(self):
        matcher = HungarianMatcher(self.matcher_config["matcher_weight_dict"], self.loss_config["use_focal_loss"], self.matcher_config["alpha"], self.matcher_config["gamma"])
        return DFINECriterion(matcher, self.loss_config["loss_weight_dict"], self.loss_config["losses"], self.loss_config["alpha"], self.loss_config["gamma"], self.num_classes, self.reg_max) if self.model_name_type == "dfine" else DEIMCriterion(matcher, self.loss_config["loss_weight_dict"], self.loss_config["losses"], self.loss_config["alpha"], self.loss_config["gamma"], self.num_classes, self.reg_max)
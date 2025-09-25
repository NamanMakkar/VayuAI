# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from vajra.nn.modules import FrozenBatchNorm2d
from vajra.utils import LOGGER

class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=1.0):
        super(LearnableAffineBlock, self).__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, groups=1, padding="", use_act=True, use_lab=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups=groups
        self.padding=padding
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_c, out_c, kernel_size, stride, groups=groups, bias=False)
            )
        else:
            self.conv = nn.Conv2d(
                in_c, out_c, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False
            )
        self.bn = nn.BatchNorm2d(out_c)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x
    
    def get_module_info(self):
        return "HGNetV2_ConvBNAct", f"[{self.in_c}, {self.out_c}, {self.kernel_size}, {self.stride}, {self.groups}, {self.padding}, {self.use_act}, {self.use_lab}]"

class LightConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, use_lab=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.use_lab = use_lab
        self.conv1 = ConvBNAct(in_c, out_c, kernel_size=1, stride=1, use_act=False, use_lab=use_lab) # pw
        self.conv2 = ConvBNAct(out_c, out_c, kernel_size, stride=1, groups=out_c, use_act=True, use_lab=use_lab) # dw

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_module_info(self):
        return "HGNetV2_LightConvBNAct", f"[{self.in_c}, {self.out_c}, {self.kernel_size}, {self.use_lab}]"

class StemBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c, use_lab=False):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.use_lab = use_lab

        self.stem1 = ConvBNAct(in_c, mid_c, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_c, out_c, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)
    
    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x
    
    def get_module_info(self):
        return "HGNetV2_StemBlock", f"[{self.in_c}, {self.mid_c}, {self.out_c}, {self.use_lab}]"
    
class EseModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)
    
    def get_module_info(self):
        return "HGNetV2_EseModule", f"[{self.channels}]"
    
class HG_Block(nn.Module):
    def __init__(self, in_c, mid_c, out_c, layer_num, kernel_size=3, residual=False, light_block=False, use_lab=False, agg="ese", drop_path=0.0):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.layer_num = layer_num
        self.kernel_size = kernel_size
        self.light_block = light_block
        self.use_lab = use_lab
        self.agg_type = agg
        self.drop_path_prob = drop_path

        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_c if i == 0 else mid_c,
                        mid_c,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_c if i == 0 else mid_c,
                        mid_c,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab
                    )
                )
        total_c = in_c + layer_num * mid_c
        if agg == "se":
            aggregation_squeeze_conv = ConvBNAct(
                total_c,
                out_c // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab
            )
            aggregation_excitation_conv = ConvBNAct(
                out_c // 2,
                out_c,
                kernel_size=1,
                stride=1,
                use_lab=use_lab
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv
            )
        else:
            aggregation_conv = ConvBNAct(
                total_c,
                out_c,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_c)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]

        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        x = self.aggregation(x)

        if self.residual:
            x = self.drop_path(x) + identity
        return x
    
    def get_module_info(self):
        return "HGBlock", f"[{self.in_c}, {self.mid_c}, {self.out_c}, {self.layer_num}, {self.kernel_size}, {self.residual}, {self.light_block}, {self.use_lab}, {self.agg_type}, {self.drop_path_prob}]"
    
class HG_Stage(nn.Module):
    def __init__(self, in_c, mid_c, out_c, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg="se", drop_path=0.0):
        super().__init__()
        self.downsample = downsample

        if downsample:
            self.downsample = ConvBNAct(
                in_c, in_c, kernel_size=3, stride=2, groups=in_c, use_act=False, use_lab=use_lab
            )
        else:
            self.downsample = nn.Identity()

        block_list = []
        for i in range(block_num):
            block_list.append(
                HG_Block(
                    in_c if i == 0 else out_c,
                    mid_c,
                    out_c,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x
    
class HGNetV2(nn.Module):
    arch_configs = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth",
        },

        "B1": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth",
        },

        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth",
        },

        "B3": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 3],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth",
        },

        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
               "stage1": [48, 48, 128, 1, False, False, 3, 6],
               "stage2": [128, 96, 512, 1, True, False, 3, 6],
               "stage3": [512, 192, 1024, 3, True, True, 5, 6],
               "stage4": [1024, 384, 2048, 1, True, True, 5, 6], 
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth",
        },

        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth",
        },

        "B6": {
            "stem_channels": [3, 48, 96],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth",
        },
    }

    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3], freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, local_model_dir="weights/hgnetv2/"):
        super().__init__()
        self.name = name
        self.use_lab = use_lab
        self.return_idx = return_idx
        self.freeze_stem_only_bool = freeze_stem_only
        self.freeze_at_idx = freeze_at
        self.freeze_norm_bool = freeze_norm
        self.pretrained_bool = pretrained
        self.local_model_dir_str = local_model_dir

        stem_channels = self.arch_configs[name]["stem_channels"]
        stage_config = self.arch_configs[name]["stage_config"]
        download_url = self.arch_configs[name]["url"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        self.stem = StemBlock(in_c=stem_channels[0], mid_c=stem_channels[1], out_c=stem_channels[2], use_lab=use_lab)

        self.stages = nn.ModuleList()

        for i, k in enumerate(stage_config):
            in_c, mid_c, out_c, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]

            self.stages.append(
                HG_Stage(
                    in_c, mid_c, out_c, block_num, layer_num, downsample, light_block, kernel_size, use_lab,
                )
            )
        
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

            try:
                os.makedirs(local_model_dir, exist_ok=True)

                model_path = os.path.join(local_model_dir, f"PPHGNetV2_{name}_stage1.pth")
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location="cpu")
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                
                else:
                    is_main_process = (
                        not torch.distributed.is_available()
                        or not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0
                    )

                    if is_main_process:
                        LOGGER.info("✅ Downloading pretrained model...")

                        try:
                            state = torch.hub.load_state_dict_from_url(
                                download_url, map_location="cpu", model_dir=local_model_dir, progress=True
                            )
                            LOGGER.info(f"✅ Successfully downloaded and loaded stage1 {name} HGNetV2.")
                        except Exception as e:
                            LOGGER.info(f"❌ Download failed: {e}" + RESET)
                            LOGGER.info(GREEN + f"Please manually download from {download_url} to {local_model_dir}" + RESET)
                            raise
                    
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.barrier()
                        if not is_main_process:
                            state = torch.load(model_path, map_location="cpu")
                
                self.load_state_dict(state)

            except Exception as e:
                LOGGER.info("❌ CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                LOGGER.info(f"Error: {str(e)}")
                LOGGER.info(GREEN + f"Please download manually from {download_url} to {local_model_dir}" + RESET)
                raise

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m
    
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
    
    def get_module_info(self):
        return "HGNetV2Backbone", f"[{self.name}, {self.use_lab}, {self.return_idx}, {self.freeze_stem_only_bool}, {self.freeze_at_idx}, {self.freeze_norm_bool}, {self.pretrained_bool}, {self.local_model_dir_str}]"
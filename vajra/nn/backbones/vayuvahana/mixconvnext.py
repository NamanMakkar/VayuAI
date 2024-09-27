# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torch.nn as nn
import torch.nn.functional as F
from vajra.nn.modules import Conv, ConvBNAct, Linear, MixConvNeXtEncoder, SDTAEncoderBNHS_GRN, LayerNorm, PositionalEncodingFourier, trunc_normal_
from vajra.nn.head import Classification, Detection, PoseDetection, Segementation, OBBDetection
from vajra.utils import LOGGER

class MixConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 1, 1, 1], global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[[3], [5,7], [7,9], [9,11,13,15]], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 3, 4, 5], **kwargs):
        super().__init__()
        for g in global_block_type:
            assert g in ['None', 'SDTA_BN_HS']

        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            ConvBNAct(in_chans, dims[0], kernel_size=4, stride=4, bias=False, act=None),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                Conv(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA_BN_HS':
                        stage_blocks.append(SDTAEncoderBNHS_GRN(dim=dims[i], drop_path=dp_rates[cur + j],
                                                            expan_ratio=expan_ratio, scales=d2_scales[i],
                                                            use_pos_emb=use_pos_embd_xca[i],
                                                            num_heads=heads[i]))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(MixConvNeXtEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                        expand_ratio=expan_ratio, dwconv_kernels=kernel_sizes[i]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        self.norm = nn.BatchNorm2d(dims[-1])

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(kwargs["classifier_dropout"])

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x).mean([-2, -1])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_dropout(x)
        return x

def build_mixconvnext(in_channels=3, 
                      num_classes=1000, 
                      size="xx_small",
                      task="classify", 
                      verbose=False):

    mixconvnext_config = {
        "xx_small": {
            "depths": [3, 3, 9, 3],
            "dims": [24, 48, 88, 168],
            "heads": [8, 8, 8, 8],
            "d2_scales": [2, 3, 4, 5],
            "use_pos_embed_xca": [False, True, False, False]
        },
        "x_small": {
            "depths": [3, 3, 9, 3],
            "dims": [32, 64, 100, 192],
            "heads": [8, 8, 8, 8],
            "d2_scales": [2, 3, 4, 5],
            "use_pos_embed_xca": [False, True, False, False]
        },
        "small": {
            "depths": [3, 3, 9, 3],
            "dims": [48, 96, 160, 304],
            "heads": [8, 8, 8, 8],
            "d2_scales": [2, 3, 4, 5],
            "use_pos_embed_xca": [False, True, False, False]
        },
        "medium":{
            "depths": [3, 3, 9, 3],
            "dims": [64, 128, 256, 512],
            "heads": [8, 8, 8, 8],
            "d2_scales": [2, 3, 4, 5],
            "use_pos_embed_xca": [False, True, False, False]
        },
        "large":{
            "depths": [3, 3, 9, 3],
            "dims": [80, 160, 288, 584],
            "heads": [8, 8, 8, 8],
            "d2_scales": [2, 3, 4, 5],
            "use_pos_embed_xca": [False, True, False, False]
        }
    }

    depths = mixconvnext_config[size]["depths"]
    dims = mixconvnext_config[size]["dims"]
    heads = mixconvnext_config[size]["heads"]
    d2_scales = mixconvnext_config[size]["d2_scales"]
    use_pos_embed_xca = mixconvnext_config[size]["use_pos_embed_xca"]

    if task == "classify":
        model = MixConvNeXt(in_channels, num_classes, depths, dims, heads=heads, use_pos_embd_xca=use_pos_embed_xca, d2_scales=d2_scales)
        head = Linear(dims[-1], num_classes)

        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        layers = []
        layers.append(model)
        layers.append(head)

        mixconvnext = nn.Sequential(*layers)
        np_model += np_head

        if verbose:
            LOGGER.info(f"Task: classify; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info(f"Building EdgeNeXt-BNHS ... \n\n")
            idx_counter = 0
            for i, (name, module) in enumerate(model.named_children()):
                if name in ["downsample_layers", "stages"]:
                    for idx, submodule in enumerate(module):
                        if isinstance(submodule, nn.Sequential):
                            for seq_idx, seq_module in enumerate(submodule):
                                if hasattr(seq_module, 'get_module_info'):
                                    np = sum(x.numel() for x in seq_module.parameters())
                                    module_info, args_info = seq_module.get_module_info()
                                    LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                                    idx_counter += 1
                        
                        else:
                            if hasattr(submodule, 'get_module_info'):
                                np = sum(x.numel() for x in submodule.parameters())
                                module_info, args_info = submodule.get_module_info()
                                LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                                idx_counter += 1

                else:
                    if hasattr(module, 'get_module_info'):
                        np = sum(x.numel() for x in module.parameters())
                        module_info, args_info = module.get_module_info()
                        LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                        idx_counter += 1

            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nModel Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"MixConvNeXt {size} Task: classify; Total Parameters: {np_model}\n\n")

        return mixconvnext, layers, np_model

    return
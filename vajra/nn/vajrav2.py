# Vayuvahana Technologies Private Limited, AGPL-3.0 License
# In experimental stages

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from vajra.checks import check_suffix, check_requirements
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.modules import LDConv, CBFuse, CBLinear, VajraV1AttentionBhag7, ConvMakkarNorm, SCDown, VajraSPPModule, VajraRepViTBlock, VajraV1MerudandaBhag15, VajraV1MerudandaX, VajraV1AttentionBhag6, VajraV1MerudandaBhag9, VajraV1MerudandaBhag11, VajraV1MerudandaBhag17, VajraDownsampleV2, VajraV1MerudandaBhag7, SPPFRepViT, AttentionBlockV2, SanlayanSPPFAttentionV7, VajraV1MerudandaBhag10, VajraV1AttentionBhag12, VajraV1MerudandaBhag12, VajraV1MerudandaBhag14, VajraV1AttentionBhag3, VajraV2AttentionBhag2, VajraV2MerudandaBhag14, VajraV2MerudandaBhag15, VajraV2MerudandaBhag13, VajraV1AttentionBhag11, VajraV1AttentionBhag1, VajraV1MerudandaBhag8, VajraV1AttentionBhag10, VajraV1MerudandaBhag8, VajraV1MerudandaBhag5, VajraV1MerudandaBhag7, VajraV2Stambh, VajraV2MerudandaBhag10, ADown, VajraStemBlock, VajraStambhV4, VajraV2AttentionBhag1, VajraV1MerudandaBhag6, VajraV1AttentionBhag9, VajraV1AttentionBhag5, VajraV1AttentionBhag2, VajraV1MakkarNormMerudandaBhag1, SanlayanSPPFAttentionV6, VajraV1MerudandaBhag4, VajraV2MerudandaBhag9, VajraV2MerudandaBhag11, VajraV2MerudandaBhag12, VajraV2MerudandaBhag8, VajraV2MerudandaBhag7, SanlayanSPPFAttentionV3, SanlayanSPPFAttentionV4, SanlayanSPPFAttentionV5, SanlayanSPPFVajraMerudandaV2, SanlayanSPPFAttentionV2, SanlayanSPPFVajraMerudanda, VajraV2StemBlock, VajraV2MerudandaBhag4, VajraV2MerudandaBhag6, VajraV2MerudandaBhag5, SanlayanSPPFAttentionBhag2, Upsample, VajraV1Merudanda, VajraV2MerudandaBhag3, VajraV1MerudandaBhag2, VajraDownsample, VajraStambh, RepVGGDW, SPPF, Concatenate, VajraStambhV2, VajraMerudandaV2Bhag1, VajraV2MerudandaBhag1, VajraV2MerudandaBhag2, VajraGrivaV2Bhag1, VajraGrivaV2Bhag2, ADown, Bottleneck, MerudandaDW, VajraMerudandaBhag1, VajraMerudandaMS, VajraMerudandaBhag7, VajraMerudandaBhag2, VajraMerudandaBhag3, VajraMerudandaBhag5, VajraGrivaBhag1, VajraGrivaBhag2, VajraGrivaBhag3, VajraGrivaBhag4, VajraMerudandaBhag4, VajraMerudandaBhag7, VajraMBConvBlock, VajraConvNeXtBlock, Sanlayan, ChatushtayaSanlayan, ConvBNAct, MaxPool, ImagePoolingAttention, VajraWindowAttnBottleneck, VajraV2BottleneckBlock, AttentionBottleneck, AttentionBottleneckV3, AttentionBottleneckV2, AttentionBottleneckV4, AttentionBottleneckV6, SanlayanSPPFAttention, DvayaSanlayan, VajraGrivaV2
from vajra.nn.head import Detection, OBBDetection, Segmentation, Classification, PoseDetection, WorldDetection, Panoptic
from vajra.nn.window_attention import VajraSwinTransformerLayer, VajraV1SwinTransformerBlockV4
from vajra.utils import LOGGER, HYPERPARAMS_CFG_DICT, HYPERPARAMS_CFG_KEYS
from vajra.utils.torch_utils import model_info, initialize_weights, fuse_conv_and_bn, time_sync, intersect_dicts, scale_img

try:
    import thop
except ImportError:
    thop = None

class VajraV2ModelNSM(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[2, 2, 2, 2, 2, 2, 2, 2],
                 inner_block_list=[False, False, True, True, False, False, False, True],
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, [5, 7], -1, [5, -1], -1, -1, [3, -1], -1, -1, [11, -1], -1, -1, [9, -1], -1, [13, 16, 19]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraV1MerudandaBhag6(channels_list[1], channels_list[2], num_repeats[0], 1, True, inner_block=inner_block_list[0], expansion_ratio=0.25, rep_bottleneck=True) #VajraV1MerudandaBhag15(channels_list[1], channels_list[2], 4, True, 0.25, use_mlp=True, kernel_size=5) #VajraV1MerudandaBhag10(channels_list[1], channels_list[2], 2*num_repeats[0], True, expansion_ratio=0.25, inner_block=False, mlp_res_vit=False) #VajraV1MerudandaBhag8(channels_list[1], channels_list[2], num_repeats[0], kernel_size=1, shortcut=True, expansion_ratio=0.25, inner_block=inner_block_list[0]) # stride 4
        self.conv1 = ConvBNAct(channels_list[2], channels_list[2], 2, 3)
        self.vajra_block2 = VajraV1MerudandaBhag6(channels_list[2], channels_list[3], num_repeats[1], 1, True, inner_block=inner_block_list[1], expansion_ratio=0.25, rep_bottleneck=True) #VajraV1MerudandaBhag15(channels_list[2], channels_list[3], 8, True, 0.25, use_mlp=False, kernel_size=5) #VajraV1MerudandaBhag10(channels_list[2], channels_list[3], 2*num_repeats[1], True, expansion_ratio=0.25, inner_block=False, mlp_res_vit=False) #VajraV1MerudandaBhag8(channels_list[2], channels_list[3], num_repeats[1], kernel_size=1, shortcut=True, expansion_ratio=0.25, inner_block=inner_block_list[1]) # stride 8
        self.conv2 = ConvBNAct(channels_list[3], channels_list[3], 2, 3)
        self.vajra_block3 = VajraV1MerudandaBhag6(channels_list[3], channels_list[3], num_repeats[2], 1, True, inner_block=inner_block_list[2], rep_bottleneck=True) #VajraV1MerudandaBhag17(channels_list[3], channels_list[3], num_repeats[2], 1, True, 0.5, grid_size=4) #VajraV1MerudandaBhag17(channels_list[3], channels_list[3], num_repeats[2], 1, True) # #VajraV1AttentionBhag1(channels_list[3], channels_list[3], num_blocks=2*num_repeats[2], mlp_ratio=1.5, area=4, lx=True) # stride 16
        self.conv3 = ConvBNAct(channels_list[3], channels_list[4], 2, 3) #VajraDownsampleV2(channels_list[3], channels_list[4] // 2, channels_list[4])
        self.vajra_block4 = VajraV1MerudandaBhag6(channels_list[4], channels_list[4], num_repeats[3], 1, True, inner_block=inner_block_list[3], rep_bottleneck=True) #VajraV1MerudandaBhag17(channels_list[4], channels_list[4], num_repeats[3], 1, True, 0.5, grid_size=1) #VajraV1MerudandaBhag17(channels_list[4], channels_list[4], num_repeats[3], 1, True) #VajraV1AttentionBhag1(channels_list[4], channels_list[4], num_blocks=2*num_repeats[3], mlp_ratio=1.5, area=1, lx=True) # stride 32
        self.sppf_attn = VajraV1AttentionBhag6(channels_list[4], channels_list[4], num_blocks=num_repeats[3], mlp_ratio=2.0)
        # Neck
        self.upsample1 = Upsample(2, "nearest")
        self.concat1 = Concatenate(in_c=[channels_list[4], channels_list[4]], dimension=1)
        self.vajra_neck1 = VajraV1MerudandaBhag6(in_c=channels_list[3] + channels_list[4], out_c=channels_list[6], num_blocks=num_repeats[4], kernel_size=1, shortcut=True, inner_block=inner_block_list[4], rep_bottleneck=True)

        self.upsample2 = Upsample(2, "nearest")
        self.concat2 = Concatenate(in_c=[channels_list[6], channels_list[2]], dimension=1)
        self.vajra_neck2 = VajraV1MerudandaBhag6(in_c=channels_list[6] + channels_list[3], out_c=channels_list[8], num_blocks=num_repeats[5], kernel_size=1, shortcut=True, inner_block=inner_block_list[5], rep_bottleneck=True)

        self.neck_conv1 = ConvBNAct(channels_list[8], channels_list[9], 2, 3)
        self.concat3 = Concatenate(in_c=[channels_list[6], channels_list[9]], dimension=1)
        self.vajra_neck3 = VajraV1MerudandaBhag6(in_c=channels_list[6] + channels_list[9], out_c=channels_list[10], num_blocks=num_repeats[6], kernel_size=1, shortcut=True, inner_block=inner_block_list[6], rep_bottleneck=True)

        self.neck_conv2 = ConvBNAct(channels_list[10], channels_list[11], 2, 3)
        self.concat4 = Concatenate(in_c=[channels_list[11], channels_list[4]], dimension=1)
        self.vajra_neck4 = VajraV1MerudandaBhag6(in_c=channels_list[4] + channels_list[11], out_c=channels_list[12], num_blocks=num_repeats[7], kernel_size=1, shortcut=True, inner_block=inner_block_list[7], rep_bottleneck=True)

    def forward(self, x):
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        conv1 = self.conv1(vajra1)
        vajra2 = self.vajra_block2(conv1)

        conv2 = self.conv2(vajra2)
        vajra3 = self.vajra_block3(conv2)

        conv3 = self.conv3(vajra3)
        vajra4 = self.sppf_attn(self.vajra_block4(conv3))
        
        # Neck
        neck_upsample1 = self.upsample1(vajra4)
        concat_neck1 = self.concat1([vajra3, neck_upsample1])
        vajra_neck1 = self.vajra_neck1(concat_neck1)

        neck_upsample2 = self.upsample2(vajra_neck1)
        concat_neck2 = self.concat2([vajra2, neck_upsample2])
        vajra_neck2 = self.vajra_neck2(concat_neck2)

        neck_conv1 = self.neck_conv1(vajra_neck2)
        concat_neck3 = self.concat3([vajra_neck1, neck_conv1])
        vajra_neck3 = self.vajra_neck3(concat_neck3)

        neck_conv2 = self.neck_conv2(vajra_neck3)
        concat_neck4 = self.concat4([vajra4, neck_conv2])
        vajra_neck4 = self.vajra_neck4(concat_neck4)

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class VajraV2ModelLX(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[2, 2, 2, 2, 2, 2, 2, 2],
                 inner_block_list=[False, False, True, True, False, False, False, True],
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, [5, 7], -1, [5, -1], -1, -1, [3, -1], -1, -1, [11, -1], -1, -1, [9, -1], -1, [13, 16, 19]]
        # Backbone
        self.stem1 = ConvBNAct(in_channels, channels_list[0], 2, 3) #VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.stem2 = ConvBNAct(channels_list[0], channels_list[1], 2, 3)
        self.vajra_block1 = VajraV1MerudandaBhag6(channels_list[1], channels_list[1], 3, 1, True, inner_block=False, expansion_ratio=0.5, rep_bottleneck=False, main_bottleneck_exp=1.0) #VajraV1MerudandaBhag15(channels_list[1], channels_list[2], 4, True, 0.25, use_mlp=True, kernel_size=5) #VajraV1MerudandaBhag10(channels_list[1], channels_list[2], 2*num_repeats[0], True, expansion_ratio=0.25, inner_block=False, mlp_res_vit=False) #VajraV1MerudandaBhag8(channels_list[1], channels_list[2], num_repeats[0], kernel_size=1, shortcut=True, expansion_ratio=0.25, inner_block=inner_block_list[0]) # stride 4
        self.conv1 = ConvBNAct(channels_list[1], channels_list[2], 2, 3)
        self.vajra_block2 = VajraV1MerudandaBhag6(channels_list[2], channels_list[2], 6, 1, True, inner_block=False, expansion_ratio=0.5, rep_bottleneck=False, main_bottleneck_exp=1.0) #VajraV1MerudandaBhag15(channels_list[2], channels_list[3], 8, True, 0.25, use_mlp=False, kernel_size=5) #VajraV1MerudandaBhag10(channels_list[2], channels_list[3], 2*num_repeats[1], True, expansion_ratio=0.25, inner_block=False, mlp_res_vit=False) #VajraV1MerudandaBhag8(channels_list[2], channels_list[3], num_repeats[1], kernel_size=1, shortcut=True, expansion_ratio=0.25, inner_block=inner_block_list[1]) # stride 8
        self.conv2 = SCDown(channels_list[2], channels_list[3], 3, 2)
        self.vajra_block3 = VajraV1MerudandaBhag6(channels_list[3], channels_list[3], 6, 1, True, inner_block=False, rep_bottleneck=False, main_bottleneck_exp=1.0) #VajraV1MerudandaBhag17(channels_list[3], channels_list[3], num_repeats[2], 1, True, 0.5, grid_size=4) #VajraV1MerudandaBhag17(channels_list[3], channels_list[3], num_repeats[2], 1, True) # #VajraV1AttentionBhag1(channels_list[3], channels_list[3], num_blocks=2*num_repeats[2], mlp_ratio=1.5, area=4, lx=True) # stride 16
        self.conv3 = SCDown(channels_list[3], channels_list[4], 3, 2)
        self.vajra_block4 = VajraV1MerudandaBhag15(channels_list[4], channels_list[4], 3, True, 0.5, True, 3, False) #VajraV1MerudandaBhag6(channels_list[4], channels_list[4], num_repeats[3], 1, True, inner_block=inner_block_list[3], rep_bottleneck=True) #VajraV1MerudandaBhag17(channels_list[4], channels_list[4], num_repeats[3], 1, True, 0.5, grid_size=1) #VajraV1MerudandaBhag17(channels_list[4], channels_list[4], num_repeats[3], 1, True) #VajraV1AttentionBhag1(channels_list[4], channels_list[4], num_blocks=2*num_repeats[3], mlp_ratio=1.5, area=1, lx=True) # stride 32
        self.sppf_attn = VajraV1AttentionBhag6(channels_list[4], channels_list[4], num_blocks=num_repeats[3], mlp_ratio=2.0)
        # Neck
        self.upsample1 = Upsample(2, "nearest")
        self.concat1 = Concatenate(in_c=[channels_list[4], channels_list[4]], dimension=1)
        self.vajra_neck1 = VajraV1MerudandaBhag15(channels_list[3] + channels_list[4], channels_list[6], 3, True, 0.5, False, 3, False) #VajraV1MerudandaBhag6(in_c=channels_list[3] + channels_list[4], out_c=channels_list[6], num_blocks=num_repeats[4], kernel_size=1, shortcut=True, inner_block=inner_block_list[4], rep_bottleneck=True)

        self.upsample2 = Upsample(2, "nearest")
        self.concat2 = Concatenate(in_c=[channels_list[6], channels_list[2]], dimension=1)
        self.vajra_neck2 = VajraV1MerudandaBhag6(in_c=channels_list[6] + channels_list[2], out_c=channels_list[8], num_blocks=3, kernel_size=1, shortcut=False, inner_block=False, rep_bottleneck=False, main_bottleneck_exp=1.0)

        self.neck_conv1 = ConvBNAct(channels_list[8], channels_list[9], 2, 3)
        self.concat3 = Concatenate(in_c=[channels_list[6], channels_list[9]], dimension=1)
        self.vajra_neck3 = VajraV1MerudandaBhag15(channels_list[6] + channels_list[9], channels_list[10], 3, True, 0.5, False, 3, False) #VajraV1MerudandaBhag6(in_c=channels_list[6] + channels_list[9], out_c=channels_list[10], num_blocks=num_repeats[6], kernel_size=1, shortcut=True, inner_block=inner_block_list[6], rep_bottleneck=True)

        self.neck_conv2 = SCDown(channels_list[10], channels_list[11], 3, 2)
        self.concat4 = Concatenate(in_c=[channels_list[11], channels_list[4]], dimension=1)
        self.vajra_neck4 = VajraV1MerudandaBhag15(channels_list[4] + channels_list[11], channels_list[12], 3, True, 0.5, False, 3, False) #VajraV1MerudandaBhag6(in_c=channels_list[4] + channels_list[11], out_c=channels_list[12], num_blocks=num_repeats[7], kernel_size=1, shortcut=True, inner_block=inner_block_list[7], rep_bottleneck=True)

    def forward(self, x):
        stem = self.stem2(self.stem1(x))
        vajra1 = self.vajra_block1(stem)

        conv1 = self.conv1(vajra1)
        vajra2 = self.vajra_block2(conv1)

        conv2 = self.conv2(vajra2)
        vajra3 = self.vajra_block3(conv2)

        conv3 = self.conv3(vajra3)
        vajra4 = self.sppf_attn(self.vajra_block4(conv3))
        
        # Neck
        neck_upsample1 = self.upsample1(vajra4)
        concat_neck1 = self.concat1([vajra3, neck_upsample1])
        vajra_neck1 = self.vajra_neck1(concat_neck1)

        neck_upsample2 = self.upsample2(vajra_neck1)
        concat_neck2 = self.concat2([vajra2, neck_upsample2])
        vajra_neck2 = self.vajra_neck2(concat_neck2)

        neck_conv1 = self.neck_conv1(vajra_neck2)
        concat_neck3 = self.concat3([vajra_neck1, neck_conv1])
        vajra_neck3 = self.vajra_neck3(concat_neck3)

        neck_conv2 = self.neck_conv2(vajra_neck3)
        concat_neck4 = self.concat4([vajra4, neck_conv2])
        vajra_neck4 = self.vajra_neck4(concat_neck4)

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class VajraV2CLSModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[64, 128, 256, 512, 1024],
                 num_repeats=[2, 2, 2, 2],
                 inner_block_list=[False, False, True, True, False, False, False, True]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.stem1 = ConvBNAct(in_channels, channels_list[0], stride=2, kernel_size=3)
        self.stem2 = ConvBNAct(channels_list[0], channels_list[1], stride=2, kernel_size=3)
        self.vajra_block1 = VajraV1MerudandaBhag6(channels_list[1], channels_list[2], num_blocks=num_repeats[0], shortcut=True, expansion_ratio=0.25, inner_block=inner_block_list[0]) # stride 4
        self.conv1 = ConvBNAct(channels_list[2], channels_list[2], 2, 3, groups=4)
        self.vajra_block2 = VajraV1MerudandaBhag6(channels_list[2], channels_list[3], num_repeats[1], shortcut=True, expansion_ratio=0.25, inner_block=inner_block_list[1]) # stride 8
        self.conv2 = ConvBNAct(channels_list[3], channels_list[3], 2, 3)
        self.vajra_block3 = VajraV1MerudandaBhag10(channels_list[3], channels_list[3], num_repeats[2], shortcut=True, expansion_ratio=0.5, inner_block=inner_block_list[2], mlp_res_vit=False) #VajraV1AttentionBhag2(channels_list[3], channels_list[3], num_repeats[2], area=4) # stride 16
        self.conv3 = ConvBNAct(channels_list[3], channels_list[4], 2, 3)
        self.vajra_block4 = VajraV1MerudandaBhag10(channels_list[4], channels_list[4], num_repeats[3], shortcut=True, expansion_ratio=0.5, inner_block=inner_block_list[3], mlp_res_vit=False) #VajraV1AttentionBhag2(channels_list[4], channels_list[4], num_repeats[3], area=1) # stride 32
        self.sppf_attn = VajraV1AttentionBhag5(channels_list[4], channels_list[4], num_repeats[3])

    def forward(self, x):
        stem = self.stem2(self.stem1(x))
        vajra1 = self.vajra_block1(stem)

        conv1 = self.conv1(vajra1)
        vajra2 = self.vajra_block2(conv1)

        conv2 = self.conv2(vajra2)
        vajra3 = self.vajra_block3(conv2)

        conv3 = self.conv3(vajra3)
        attn_stride_32 = self.vajra_block4(conv3)
        return attn_stride_32
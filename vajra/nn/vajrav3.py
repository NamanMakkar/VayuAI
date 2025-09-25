# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License
# In experimental stages

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from vajra.checks import check_suffix, check_requirements
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.modules import VajraStemBlock, VajraV2StemBlock, VajraStambhV2, VajraV3StemBlock, VajraStambhV3, VajraV1MerudandaBhag2, SanlayanSPPFAttention, VajraV2MerudandaBhag3, SanlayanSPPFAttentionV2, SanlayanSPPFAttentionV3, SanlayanSPPFVajraMerudanda, SanlayanSPPFVajraMerudandaV2, VajraStambh, VajraMerudandaBhag1, VajraMerudandaBhag3, VajraMerudandaBhag5, VajraMerudandaBhag6, VajraMerudandaBhag7, VajraGrivaBhag1, VajraGrivaBhag2, VajraMerudandaBhag2, VajraMerudandaBhag4, VajraMBConvBlock, VajraConvNeXtBlock, Sanlayan, SPPF, Concatenate, Upsample, SanlayanSPPF, ChatushtayaSanlayan, TritayaSanlayan, AttentionBottleneck, AttentionBottleneckV2, AttentionBottleneckV4, ConvBNAct, DepthwiseConvBNAct, MaxPool, ImagePoolingAttention, VajraWindowAttnBottleneck, VajraV2BottleneckBlock, VajraV3BottleneckBlock, ADown
from vajra.nn.head import Detection, OBBDetection, Segmentation, Classification, PoseDetection, WorldDetection, Panoptic
from vajra.utils import LOGGER, HYPERPARAMS_CFG_DICT, HYPERPARAMS_CFG_KEYS
from vajra.utils.torch_utils import model_info, initialize_weights, fuse_conv_and_bn, time_sync, intersect_dicts, scale_img
from vajra.loss import DetectionLoss, OBBLoss, SegmentationLoss, PoseLoss, ClassificationLoss

try:
    import thop
except ImportError:
    thop = None

class VajraV3Model(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[2, 2, 2, 2, 2, 2, 2, 2],
                 inner_block_list=[False, False, True, True, False, False, False, True]
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [5, 7], -1, [5, -1], -1, -1, [3, -1], -1, -1, [11, -1], -1, -1, [9, -1], -1, [13, 16, 19]]
        # Backbone
        self.stem = VajraStambhV2(in_channels, channels_list[0], channels_list[1]) #VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraV2MerudandaBhag3(channels_list[1], channels_list[2], num_repeats[0], True, 0.25, inner_block=inner_block_list[0]) # stride 4
        self.conv1 = ConvBNAct(channels_list[2], channels_list[2], 2, 3)
        self.vajra_block2 = VajraV2MerudandaBhag3(channels_list[2], channels_list[3], num_repeats[1], True, 0.25, inner_block=inner_block_list[1]) # stride 8
        self.conv2 = ConvBNAct(channels_list[3], channels_list[3], 2, 3)
        self.vajra_block3 = VajraV2MerudandaBhag3(channels_list[3], channels_list[4], num_repeats[2], True, inner_block=inner_block_list[2]) # stride 16
        self.conv3 = ConvBNAct(channels_list[4], channels_list[4], 2, 3)
        self.vajra_block4 = VajraV2MerudandaBhag3(channels_list[4], channels_list[4], 2*num_repeats[3], True, inner_block=inner_block_list[3]) # stride 32
        #self.pyramid_pool_attn_block1 = SanlayanSPPFVajraMerudanda(2*channels_list[4], channels_list[4] // 2, 1, num_repeats[3], use_sppf=True)
        #self.sppf = SPPF(channels_list[2], channels_list[2], 5)
        self.vajra_sanlayan_block = SanlayanSPPFAttentionV3(channels_list[4], channels_list[4], 1, 2*num_repeats[3]) # stride 32
        self.vajra_sanlayan_block1 = SanlayanSPPFVajraMerudandaV2(channels_list[4], channels_list[4] // 2, 1, 2*num_repeats[3], use_sppf=True, inner_block=inner_block_list[4]) # stride 8
        self.vajra_sanlayan_block2 = SanlayanSPPFVajraMerudanda(channels_list[4] // 2 + channels_list[4], int(0.75 * channels_list[4]), 1, num_repeats[3], use_sppf=True) # stride 16
        self.vajra_sanlayan_block3 = SanlayanSPPFVajraMerudanda(channels_list[4] // 2 + int(0.75 * channels_list[4]), channels_list[4], 2, num_repeats[3], use_sppf=True) # stride 32 int(0.75 * channels_list[4])
        self.vajra_sanlayan_block4 = SanlayanSPPFAttentionV2(channels_list[4] + int(0.75 * channels_list[4]), channels_list[4], 2, 2*num_repeats[3], use_sppf=True) # stride 64

    def forward(self, x):
        # Backbone
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        pool1 = self.conv1(vajra1)
        vajra2 = self.vajra_block2(pool1)

        pool2 = self.conv2(vajra2)
        vajra3 = self.vajra_block3(pool2)

        pool3 = self.conv3(vajra3)
        vajra4 = self.vajra_block4(pool3)

        #Neck
        #vajra5 = self.vajra_block5(vajra4) 
        #sppf = self.sppf(vajra1)
        vajra_sanlayan_block = self.vajra_sanlayan_block([vajra3, vajra4])
        vajra_sanlayan_block1 = self.vajra_sanlayan_block1([vajra_sanlayan_block, vajra2]) # stride 8
        vajra_sanlayan_block2 = self.vajra_sanlayan_block2([vajra_sanlayan_block1, vajra3]) # stride 16
        vajra_sanlayan_block3 = self.vajra_sanlayan_block3([vajra_sanlayan_block1, vajra_sanlayan_block2]) # stride 32
        vajra_sanlayan_block4 = self.vajra_sanlayan_block4([vajra_sanlayan_block2, vajra_sanlayan_block3]) # stride 64


        outputs = [vajra_sanlayan_block1, vajra_sanlayan_block2, vajra_sanlayan_block3, vajra_sanlayan_block4]
        return outputs

class VajraV3CLSModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[64, 128, 256, 512, 1024],
                 num_repeats=[2, 2, 2, 2],
                 inner_block_list=[False, False, True, True, False, False, False, True]
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1]
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaBhag3(channels_list[1], channels_list[2], num_repeats[0], 1, True, 0.25, False, inner_block_list[0]) # stride 4
        self.pool1 = ConvBNAct(channels_list[2], channels_list[2], 2, 3)
        self.vajra_block2 = VajraMerudandaBhag3(channels_list[2], channels_list[3], num_repeats[1], 1, True, 0.25, False, inner_block_list[1]) # stride 8
        self.pool2 = ConvBNAct(channels_list[3], channels_list[3], 2, 3)
        self.vajra_block3 = VajraMerudandaBhag3(channels_list[3], channels_list[4], num_repeats[2], 1, True, inner_block=inner_block_list[2]) # stride 16
        self.pool3 = ConvBNAct(channels_list[4], channels_list[4], 2, 3)
        self.vajra_block4 = VajraMerudandaBhag3(channels_list[4], channels_list[4], num_repeats[3], 1, True, inner_block=inner_block_list[3]) # stride 32
        self.sanlayan = Sanlayan(in_c=[channels_list[2], channels_list[3], channels_list[4], channels_list[4]], out_c=channels_list[4], stride=1, expansion_ratio=1.0)
        self.pyramid_pool = SPPF(channels_list[4], channels_list[4])
        self.attn_block = AttentionBottleneck(channels_list[4], channels_list[4], 2)

    def forward(self, x):
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        pool1 = self.pool1(vajra1)
        vajra2 = self.vajra_block2(pool1)

        pool2 = self.pool2(vajra2)
        vajra3 = self.vajra_block3(pool2)

        pool3 = self.pool3(vajra3)
        vajra4 = self.vajra_block4(pool3)
        pyramid_pool_backbone = self.pyramid_pool([vajra1, vajra2, vajra3, vajra4])
        attn_block = self.attn_block(pyramid_pool_backbone)

        return attn_block
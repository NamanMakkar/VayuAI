# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License
# In experimental stages

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from vajra.checks import check_suffix, check_requirements
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.modules import VajraStemBlock, VajraV2StemBlock, VajraStambh, SPPF, Concatenate, VajraStambhV2, VajraMerudandaV2Bhag1, VajraV2MerudandaBhag1, VajraGrivaV2Bhag1, VajraGrivaV2Bhag2, ADown, Bottleneck, MerudandaDW, VajraMerudandaBhag1, VajraMerudandaMS, VajraMerudandaBhag7, VajraMerudandaBhag2, VajraMerudandaBhag3, VajraMerudandaBhag5, VajraGrivaBhag1, VajraGrivaBhag2, VajraGrivaBhag3, VajraGrivaBhag4, VajraMerudandaBhag4, VajraMerudandaBhag7, VajraMBConvBlock, VajraConvNeXtBlock, Sanlayan, ChatushtayaSanlayan, ConvBNAct, MaxPool, ImagePoolingAttention, VajraWindowAttnBottleneck, VajraV2BottleneckBlock, AttentionBottleneck, AttentionBottleneckV3, AttentionBottleneckV2, AttentionBottleneckV4, AttentionBottleneckV6, SanlayanSPPFAttention
from vajra.nn.head import Detection, OBBDetection, Segementation, Classification, PoseDetection, WorldDetection, Panoptic
from vajra.utils import LOGGER, HYPERPARAMS_CFG_DICT, HYPERPARAMS_CFG_KEYS
from vajra.utils.torch_utils import model_info, initialize_weights, fuse_conv_and_bn, time_sync, intersect_dicts, scale_img
from vajra.loss import DetectionLoss, OBBLoss, SegmentationLoss, PoseLoss, ClassificationLoss

try:
    import thop
except ImportError:
    thop = None

class VajraV2Model(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[2, 2, 2, 2, 2, 2, 2, 2],
                 sanlayan_griva = False,
                 inner_block_list=[False, False, True, True, False, False, False, True]
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, [5, -1], -1, [3, -1], -1, -1, [11, -1], -1, -1, [9, -1], -1, [13, 16, 19]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaV2Bhag1(channels_list[1], channels_list[2], num_blocks=num_repeats[0], inner_block=inner_block_list[0], kernel_size=3) # stride 4
        self.conv1 = ConvBNAct(channels_list[2], channels_list[2], 2, 3)
        self.vajra_block2 = VajraMerudandaV2Bhag1(channels_list[2], channels_list[3], num_blocks=num_repeats[1], inner_block=inner_block_list[1], kernel_size=5) # stride 8
        self.conv2 = ConvBNAct(channels_list[3], channels_list[3], 2, 3)
        self.vajra_block3 = VajraMerudandaV2Bhag1(channels_list[3], channels_list[4], num_blocks=num_repeats[2], inner_block=inner_block_list[2], kernel_size=7) # stride 16
        self.conv3 = ConvBNAct(channels_list[4], channels_list[4], 2, 3)
        self.vajra_block4 = VajraMerudandaV2Bhag1(channels_list[4], channels_list[4], num_blocks=num_repeats[3], inner_block=inner_block_list[3], kernel_size=9) # stride 32
        self.pyramid_pool = SPPF(channels_list[4], channels_list[4])
        self.attn_block = AttentionBottleneckV6(channels_list[4], channels_list[4], 2)
        # Neck
        self.concat1 = Concatenate(in_c=[channels_list[4], channels_list[4]], dimension=1)
        self.vajra_neck1 = VajraGrivaV2Bhag2(in_c=2 * channels_list[4], out_c=channels_list[6], num_blocks=num_repeats[4], inner_block=inner_block_list[4]) if not sanlayan_griva else VajraGrivaV2Bhag1(2 * channels_list[4], channels_list[6], num_repeats[4], inner_block_list[4]) #VajraMerudandaBhag7(in_c=2 * channels_list[4], out_c=channels_list[6], num_blocks=num_repeats[4], kernel_size=1, shortcut=True, inner_block=True)

        self.concat2 = Concatenate(in_c=[channels_list[6], channels_list[3]], dimension=1)
        self.vajra_neck2 = VajraGrivaV2Bhag2(in_c=channels_list[6] + channels_list[3], out_c=channels_list[8], num_blocks=num_repeats[5], inner_block=inner_block_list[5]) #VajraMerudandaBhag7(in_c=channels_list[6] + channels_list[3], out_c=channels_list[8], num_blocks=num_repeats[5], kernel_size=1, shortcut=True, inner_block=True)

        self.neck_conv1 = ConvBNAct(channels_list[8], channels_list[9], 2, 3)
        self.concat3 = Concatenate(in_c=[channels_list[6], channels_list[9]], dimension=1)
        self.vajra_neck3 = VajraGrivaV2Bhag2(in_c=channels_list[6] + channels_list[9], out_c=channels_list[10], num_blocks=num_repeats[6], inner_block=inner_block_list[6]) #VajraMerudandaBhag7(in_c=channels_list[6] + channels_list[9], out_c=channels_list[10], num_blocks=num_repeats[6], kernel_size=1, shortcut=True, inner_block=True)

        self.neck_conv2 = ConvBNAct(channels_list[10], channels_list[11], 2, 3)
        self.concat4 = Concatenate(in_c=[channels_list[11], channels_list[4]], dimension=1)
        self.vajra_neck4 = VajraGrivaV2Bhag2(in_c=channels_list[4] + channels_list[11], out_c=channels_list[12], num_blocks=num_repeats[7], inner_block=inner_block_list[7]) if not sanlayan_griva else VajraGrivaV2Bhag1(channels_list[4] + channels_list[11], channels_list[12], num_repeats[7], inner_block_list[7]) #VajraMerudandaBhag7(in_c=channels_list[4] + channels_list[11], out_c=channels_list[12], num_blocks=num_repeats[7], kernel_size=1, shortcut=True, inner_block=True)

    def forward(self, x):
        # Backbone
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        conv1 = self.conv1(vajra1)
        vajra2 = self.vajra_block2(conv1)

        conv2 = self.conv2(vajra2)
        vajra3 = self.vajra_block3(conv2)

        conv3 = self.conv3(vajra3)
        vajra4 = self.vajra_block4(conv3)
        
        pyramidal_pool = self.pyramid_pool(vajra4)
        attn_block = self.attn_block(pyramidal_pool)
        # Neck
        _, _, H3, W3 = vajra3.shape
        neck_upsample1 = F.interpolate(attn_block, size=(H3, W3), mode="nearest")
        concat_neck1 = self.concat1([vajra3, neck_upsample1])
        vajra_neck1 = self.vajra_neck1(concat_neck1)
        vajra_neck1 = vajra_neck1 + vajra3 if self.vajra_neck1.out_c == self.vajra_block3.out_c else vajra_neck1

        _, _, H2, W2 = vajra2.shape
        neck_upsample2 = F.interpolate(vajra_neck1, size=(H2, W2), mode="nearest")
        concat_neck2 = self.concat2([vajra2, neck_upsample2])
        vajra_neck2 = self.vajra_neck2(concat_neck2)

        neck_conv1 = self.neck_conv1(vajra_neck2)
        concat_neck3 = self.concat3([vajra_neck1, neck_conv1])
        vajra_neck3 = self.vajra_neck3(concat_neck3)
        vajra_neck3 = vajra_neck3 + vajra3 if self.vajra_neck3.out_c == self.vajra_block3.out_c else vajra_neck3

        neck_conv2 = self.neck_conv2(vajra_neck3)
        concat_neck4 = self.concat4([attn_block, neck_conv2])
        vajra_neck4 = self.vajra_neck4(concat_neck4)
        vajra_neck4 = vajra_neck4 + attn_block

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class VajraV2CLSModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[64, 128, 256, 512, 1024],
                 num_repeats=[2, 2, 2, 2],
                 inner_block_list=[False, False, True, True, False, False, False, True]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1]
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaV2Bhag1(channels_list[1], channels_list[2], num_blocks=num_repeats[0], inner_block=inner_block_list[0], kernel_size=3) # stride 4
        self.conv1 = ConvBNAct(channels_list[2], channels_list[2], 2, 3)
        self.vajra_block2 = VajraMerudandaV2Bhag1(channels_list[2], channels_list[3], num_blocks=num_repeats[1], inner_block=inner_block_list[1], kernel_size=5) # stride 8
        self.conv2 = ConvBNAct(channels_list[3], channels_list[3], 2, 3)
        self.vajra_block3 = VajraMerudandaV2Bhag1(channels_list[3], channels_list[4], num_blocks=num_repeats[2], inner_block=inner_block_list[2], kernel_size=7) # stride 16
        self.conv3 = ConvBNAct(channels_list[4], channels_list[4], 2, 3)
        self.vajra_block4 = VajraMerudandaV2Bhag1(channels_list[4], channels_list[4], num_blocks=num_repeats[3], inner_block=inner_block_list[3], kernel_size=9) # stride 32
        self.pyramid_pool_attn = SanlayanSPPFAttention(2 * channels_list[4], channels_list[4], 1, 2)

    def forward(self, x):
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        conv1 = self.conv1(vajra1)
        vajra2 = self.vajra_block2(conv1)

        conv2 = self.conv2(vajra2)
        vajra3 = self.vajra_block3(conv2)

        conv3 = self.conv3(vajra3)
        vajra4 = self.vajra_block4(conv3)
        pyramid_pool_backbone = self.pyramid_pool_attn([vajra3, vajra4])

        return pyramid_pool_backbone
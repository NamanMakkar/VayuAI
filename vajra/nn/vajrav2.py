# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License
# In experimental stages

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from vajra.checks import check_suffix, check_requirements
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.modules import VajraStemBlock, VajraV2StemBlock, VajraStambh, VajraStambhV2, ADown, Bottleneck, MerudandaDW, VajraMerudandaBhag1, VajraMerudandaBhag7, VajraMerudandaBhag2, VajraMerudandaBhag3, VajraGrivaBhag1, VajraGrivaBhag2, VajraMBConvBlock, VajraConvNeXtBlock, Sanlayan, ChatushtayaSanlayan, ConvBNAct, MaxPool, ImagePoolingAttention, VajraWindowAttnBottleneck, VajraV2BottleneckBlock, AttentionBottleneck
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
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1, [1, 3, 5, -1], -1, [1, 5, 3, -1], -1, [5 + sum(num_repeats[:4]), 6 + sum(num_repeats[:5]), -1], -1, -1, [6 + sum(num_repeats[:5]), 7 + sum(num_repeats[:6]), -1], -1, -1, [7 + sum(num_repeats[:6]), 9 + sum(num_repeats[:7]), 11 + sum(num_repeats)]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.block1 = nn.Sequential(*[Bottleneck(channels_list[1], channels_list[1], True) for _ in range(num_repeats[0])]) # stride 4
        self.conv3 = ConvBNAct(channels_list[1], channels_list[2], 2, 3)
        self.block2 = nn.Sequential(*[Bottleneck(channels_list[2], channels_list[2], True) for _ in range(num_repeats[1])]) # stride 8
        self.conv4 = ConvBNAct(channels_list[2], channels_list[3], 2, 3)
        self.block3 = nn.Sequential(*[Bottleneck(channels_list[3], channels_list[3], True) for _ in range(num_repeats[2])]) # stride 16
        self.conv5 = ConvBNAct(channels_list[3], channels_list[4], 2, 3)
        self.block4 = nn.Sequential(*[MerudandaDW(channels_list[4], channels_list[4], True, 0.5, True) for _ in range(num_repeats[3])]) # stride 32
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=1, use_cbam=False, expansion_ratio=1.0)
        self.attn_block = AttentionBottleneck(channels_list[4], channels_list[4], 2)
        # Neck
        self.fusion4cbam = ChatushtayaSanlayan(in_c=channels_list[1:5], out_c=channels_list[6], use_cbam=False, expansion_ratio=1.0)
        self.vajra_neck1 = nn.Sequential(*[MerudandaDW(channels_list[6], channels_list[6], True) for _ in range(num_repeats[4])])

        self.fusion4cbam2 = ChatushtayaSanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[6]], out_c=channels_list[8], use_cbam=False, expansion_ratio=1.0)
        self.vajra_neck2 = nn.Sequential(*[Bottleneck(channels_list[8], channels_list[8], True) for _ in range(num_repeats[5])])

        self.pyramid_pool_neck1 = Sanlayan(in_c=[channels_list[4], channels_list[6], channels_list[8]], out_c=channels_list[9], stride=1, use_cbam=False, expansion_ratio=1.0)
        self.neck_conv1 = ConvBNAct(channels_list[9], channels_list[10], 2, 3)
        self.vajra_neck3 = nn.Sequential(*[Bottleneck(channels_list[10], channels_list[10], True) for _ in range(num_repeats[6])])

        self.pyramid_pool_neck2 = Sanlayan(in_c=[channels_list[6], channels_list[8], channels_list[10]], out_c=channels_list[11], stride=1, use_cbam=False, expansion_ratio=1.0)
        self.neck_conv2 = ConvBNAct(channels_list[11], channels_list[12], 2, 3)
        self.vajra_neck4 = nn.Sequential(*[Bottleneck(channels_list[12], channels_list[12], True) for _ in range(num_repeats[7])])

    def forward(self, x):
        # Backbone
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        vajra1 = self.block1(conv2)
        vajra1 = conv2 + vajra1

        pool1 = self.conv3(vajra1)
        vajra2 = self.block2(pool1)
        vajra2 = vajra2 + pool1

        pool2 = self.conv4(vajra2)
        vajra3 = self.block3(pool2)
        vajra3 = vajra3 + pool2

        pool3 = self.conv5(vajra3)
        vajra4 = self.block4(pool3)
        vajra4 = vajra4 + pool3

        pyramid_pool_backbone = self.pyramid_pool([vajra1, vajra2, vajra3, vajra4])
        attn_block = self.attn_block(pyramid_pool_backbone)
        # Neck
        fusion4 = self.fusion4cbam([vajra1, vajra2, vajra3, attn_block])
        vajra_neck1 = self.vajra_neck1(fusion4)
        vajra_neck1 = vajra_neck1 + fusion4
        #vajra_neck1 = vajra_neck1 + vajra3

        fusion4_2 = self.fusion4cbam2([vajra1, vajra3, vajra2, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_2)
        vajra_neck2 = vajra_neck2 + fusion4_2
        #vajra_neck2 = vajra_neck2 + vajra2

        pyramid_pool_neck1 = self.pyramid_pool_neck1([attn_block, vajra_neck1, vajra_neck2])
        neck_conv1 = self.neck_conv1(pyramid_pool_neck1)
        vajra_neck3 = self.vajra_neck3(neck_conv1)
        vajra_neck3 = vajra_neck3 + neck_conv1
        #vajra_neck3 = vajra_neck3 + vajra3

        pyramid_pool_neck2 = self.pyramid_pool_neck2([vajra_neck1, vajra_neck2, vajra_neck3])
        neck_conv2 = self.neck_conv2(pyramid_pool_neck2)
        vajra_neck4 = self.vajra_neck4(neck_conv2)
        vajra_neck4 = vajra_neck4 + neck_conv2
        #vajra_neck4 = vajra_neck4 + vajra4

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

"""class VajraV2Model(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[2, 2, 2, 2, 2, 2, 2, 2],
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1, [1, 3, 5, -1], -1, [1, 5, 3, -1], -1, [9, 11, -1], -1, -1, [11, 13, -1], -1, -1, [13, 16, 19]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaBhag3(channels_list[1], channels_list[1], num_repeats[0], True, 3, 0.5, False, False) # stride 4
        self.pool1 = ADown(channels_list[1], channels_list[2])
        self.vajra_block2 = VajraMerudandaBhag3(channels_list[2], channels_list[2], num_repeats[1], True, 1, 0.5, False, False) # stride 8
        self.pool2 = ADown(channels_list[2], channels_list[3])
        self.vajra_block3 = VajraMerudandaBhag3(channels_list[3], channels_list[3], num_repeats[2], 1, True, 0.5, False, True) # stride 16
        self.pool3 = ADown(channels_list[3], channels_list[4])
        self.vajra_block4 = VajraMerudandaBhag3(channels_list[4], channels_list[4], num_repeats[3], 1, True, 0.5, False, True) # stride 32
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2, use_cbam=False, expansion_ratio=1.0)
        self.attn_block = AttentionBottleneck(channels_list[4], channels_list[4], 2)
        # Neck
        self.fusion4cbam = ChatushtayaSanlayan(in_c=channels_list[1:5], out_c=channels_list[6], use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck1 = VajraMerudandaBhag3(channels_list[6] // 2, channels_list[6], num_repeats[4], 1, True, 0.5, False)

        self.fusion4cbam2 = ChatushtayaSanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[6]], out_c=channels_list[8], use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck2 = VajraMerudandaBhag3(channels_list[8] // 2, channels_list[8], num_repeats[5], 1, True, 0.5, False)

        self.pyramid_pool_neck1 = Sanlayan(in_c=[channels_list[4], channels_list[6], channels_list[8]], out_c=channels_list[10], stride=1, use_cbam=False, expansion_ratio=0.5)
        self.neck_conv1 = ADown(channels_list[10] // 2, channels_list[10] // 2)
        self.vajra_neck3 = VajraMerudandaBhag3(channels_list[10] // 2, channels_list[10], num_repeats[6], 1, True, 0.5, False, True) #VajraGrivaBhag1(channels_list[10], num_repeats[6], 1, 0.5, False)

        self.pyramid_pool_neck2 = Sanlayan(in_c=[channels_list[6], channels_list[8], channels_list[10]], out_c=channels_list[12], stride=1, use_cbam=False, expansion_ratio=0.5)
        self.neck_conv2 = ADown(channels_list[12] // 2, channels_list[12] // 2)
        self.vajra_neck4 = VajraMerudandaBhag3(channels_list[12] // 2, channels_list[12], num_repeats[7], 1, True, 0.5, False, True) #VajraGrivaBhag1(channels_list[12], num_repeats[7], 1, 0.5, False)

    def forward(self, x):
        # Backbone
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
        # Neck
        fusion4 = self.fusion4cbam([vajra1, vajra2, vajra3, attn_block])
        vajra_neck1 = self.vajra_neck1(fusion4)
        vajra_neck1 = vajra_neck1 + vajra3

        fusion4_2 = self.fusion4cbam2([vajra1, vajra3, vajra2, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_2)
        vajra_neck2 = vajra_neck2 + vajra2

        pyramid_pool_neck1 = self.pyramid_pool_neck1([attn_block, vajra_neck1, vajra_neck2])
        neck_conv1 = self.neck_conv1(pyramid_pool_neck1)
        vajra_neck3 = self.vajra_neck3(neck_conv1)
        vajra_neck3 = vajra_neck3 + vajra3

        pyramid_pool_neck2 = self.pyramid_pool_neck2([vajra_neck1, vajra_neck2, vajra_neck3])
        neck_conv2 = self.neck_conv2(pyramid_pool_neck2)
        vajra_neck4 = self.vajra_neck4(neck_conv2)
        vajra_neck4 = vajra_neck4 + vajra4

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs
"""
        
"""class VajraV2Model(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[3, 6, 6, 3, 3, 3, 3, 3],
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], [1, 3, 5, -1], -1, [1, 5, 3, -1], -1, [8, 10, -1], -1, [10, 12, -1], -1, [12, 14, 16]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaBhag7(channels_list[1], channels_list[1], num_repeats[0], True, 3, False, 0.5, False) # stride 4
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraMerudandaBhag7(channels_list[1], channels_list[2], num_repeats[1], True, 3, False, 0.5, False) # stride 8
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraMerudandaBhag7(channels_list[2], channels_list[3], num_repeats[2], True, 3, True, 0.5, False) # stride 16
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraMerudandaBhag7(channels_list[3], channels_list[4], num_repeats[3], True, 3, True, 0.5, False) # stride 32
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2, use_cbam=False, expansion_ratio=1.0)

        # Neck
        self.fusion4cbam = ChatushtayaSanlayan(in_c=channels_list[1:5], out_c=channels_list[6], use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck1 = VajraGrivaBhag1(channels_list[6], num_repeats[4], 1, 0.5, False, True)

        self.fusion4cbam2 = ChatushtayaSanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[6]], out_c=channels_list[8], use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck2 = VajraGrivaBhag1(channels_list[8], num_repeats[5], 1, 0.5, False)

        self.fusion4cbam3 = ChatushtayaSanlayan(in_c=[channels_list[4], channels_list[6], channels_list[1], channels_list[8]], out_c=channels_list[8], use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck3 = VajraGrivaBhag1(channels_list[8], num_repeats[5], 1, 0.5, False)

        self.pyramid_pool_neck1 = Sanlayan(in_c=[channels_list[4], channels_list[6], channels_list[8]], out_c=channels_list[10], stride=2, use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck4 = VajraGrivaBhag1(channels_list[10], num_repeats[6], 1, 0.5, False, True)

        self.pyramid_pool_neck2 = Sanlayan(in_c=[channels_list[6], channels_list[8], channels_list[10]], out_c=channels_list[12], stride=2, use_cbam=False, expansion_ratio=0.5)
        self.vajra_neck5 = VajraGrivaBhag1(channels_list[12], num_repeats[7], 1, 0.5, False, True)
    
    def forward(self, x):
        # Backbone
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        pool1 = self.pool1(vajra1)
        vajra2 = self.vajra_block2(pool1)

        pool2 = self.pool2(vajra2)
        vajra3 = self.vajra_block3(pool2)

        pool3 = self.pool3(vajra3)
        vajra4 = self.vajra_block4(pool3)
        pyramid_pool_backbone = self.pyramid_pool([vajra1, vajra2, vajra3, vajra4])

        # Neck
        fusion4 = self.fusion4cbam([vajra1, vajra2, vajra3, pyramid_pool_backbone])
        vajra_neck1 = self.vajra_neck1(fusion4)
        vajra_neck1 = vajra_neck1 + vajra3

        fusion4_2 = self.fusion4cbam2([vajra1, vajra3, vajra2, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_2)
        vajra_neck2 = vajra_neck2 + vajra2

        pyramid_pool_neck1 = self.pyramid_pool_neck1([pyramid_pool_backbone, vajra_neck1, vajra_neck2])
        vajra_neck4 = self.vajra_neck4(pyramid_pool_neck1)
        vajra_neck4 = vajra_neck4 + vajra3

        pyramid_pool_neck2 = self.pyramid_pool_neck2([vajra_neck1, vajra_neck2, vajra_neck4])
        vajra_neck5 = self.vajra_neck5(pyramid_pool_neck2)
        vajra_neck5 = vajra_neck5 + vajra4

        outputs = [vajra_neck2, vajra_neck4, vajra_neck4]
        return outputs"""

class VajraV2CLSModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[64, 128, 256, 512, 1024],
                 num_repeats=[3, 6, 6, 3]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1]
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraV2BottleneckBlock(channels_list[1], channels_list[1], num_repeats[0], 1, True, 3, False) # stride 4
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraV2BottleneckBlock(channels_list[1], channels_list[2], num_repeats[1], 1, True, 3, False) # stride 8
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraV2BottleneckBlock(channels_list[2], channels_list[3], num_repeats[2], 1, True, 3, False) # stride 16
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraV2BottleneckBlock(channels_list[3], channels_list[4], num_repeats[3], 1, True, 3, False) # stride 32
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2)

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

        return pyramid_pool_backbone
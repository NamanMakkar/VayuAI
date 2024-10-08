# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License
# In experimental stages

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from vajra.checks import check_suffix, check_requirements
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.modules import VajraStemBlock, VajraV2StemBlock, VajraV3StemBlock, VajraDownsampleStem, VajraBottleneckBlock, VajraEfficientBottleneckBlock, VajraMBConvBlock, VajraConvNeXtBlock, PyramidalPoolCBAM, Fusion4CBAM, ConvBNAct, MaxPool, ImagePoolingAttention, VajraWindowAttnBottleneck, VajraV2BottleneckBlock, VajraV3BottleneckBlock
from vajra.nn.head import Detection, OBBDetection, Segementation, Classification, PoseDetection, WorldDetection, Panoptic
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
                 num_repeats=[3, 6, 6, 3, 3, 3, 3, 3]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], [1, 3, 5, -1], -1, [1, 5, 3, -1], -1, [8, 10, -1], -1, [10, 12, -1], -1, [12, 14, 16]]
        # Backbone
        self.stem = VajraStemBlock(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraBottleneckBlock(channels_list[1], channels_list[1], num_repeats[0], True, 3, False)
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraBottleneckBlock(channels_list[1], channels_list[2], num_repeats[1], True, 3, False)
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraBottleneckBlock(channels_list[2], channels_list[3], num_repeats[2], True, 3, False)
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraBottleneckBlock(channels_list[3], channels_list[4], num_repeats[3], True, 3, False)
        self.pyramid_pool = PyramidalPoolCBAM(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2, use_cbam=False)

        # Neck
        self.fusion4cbam = Fusion4CBAM(in_c=channels_list[1:5], out_c=channels_list[5], use_cbam=False)
        self.vajra_neck1 = VajraBottleneckBlock(channels_list[5], channels_list[6], num_repeats[4], False, 1, False)

        self.fusion4cbam_1 = Fusion4CBAM(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[6]], out_c=channels_list[7], use_cbam=False)
        self.vajra_neck2 = VajraBottleneckBlock(channels_list[7], channels_list[8], num_repeats[5], False, 1, False)

        self.pyramid_pool_neck1 = PyramidalPoolCBAM(in_c=[channels_list[4], channels_list[6], channels_list[8]], out_c=channels_list[9], stride=2, use_cbam=False)
        self.vajra_neck3 = VajraBottleneckBlock(channels_list[9], channels_list[10], num_repeats[6], False, 1, False)

        self.pyramid_pool_neck2 = PyramidalPoolCBAM(in_c=[channels_list[6], channels_list[8], channels_list[10]], out_c=channels_list[11], stride=2, use_cbam=False)
        self.vajra_neck4 = VajraBottleneckBlock(channels_list[11], channels_list[12], num_repeats[7], False, 1, False)

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

        fusion4_2 = self.fusion4cbam_1([vajra1, vajra3, vajra2, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_2)

        pyramid_pool_neck1 = self.pyramid_pool_neck1([pyramid_pool_backbone, vajra_neck1, vajra_neck2])
        vajra_neck3 = self.vajra_neck3(pyramid_pool_neck1)

        pyramid_pool_neck2 = self.pyramid_pool_neck2([vajra_neck1, vajra_neck2, vajra_neck3])
        vajra_neck4 = self.vajra_neck4(pyramid_pool_neck2)

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class VajraV3CLSModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[64, 128, 256, 512, 1024],
                 num_repeats=[3, 6, 6, 3]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1]
        self.stem = VajraDownsampleStem(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraV3BottleneckBlock(channels_list[1], channels_list[1], num_repeats[0], 3, True, 3, False) # stride 4
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraV3BottleneckBlock(channels_list[1], channels_list[2], num_repeats[1], 3, True, 3, False) # stride 8
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraV3BottleneckBlock(channels_list[2], channels_list[3], num_repeats[2], 3, True, 3, False) # stride 16
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraV3BottleneckBlock(channels_list[3], channels_list[4], num_repeats[3], 3, True, 3, False) # stride 32
        self.pyramid_pool = PyramidalPoolCBAM(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2)

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
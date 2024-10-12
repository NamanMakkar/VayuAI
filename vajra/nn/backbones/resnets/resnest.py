# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import BottleneckResNetSplitAttention, Linear, ConvBNAct, Sanlayan, ChatushtayaSanlayan, VajraMerudandaBhag1, MaxPool, AdaptiveAvgPool2D
from vajra.nn.head import Classification, Detection, PoseDetection, OBBDetection, Segementation
from vajra.utils import LOGGER
from vajra.ops import make_divisible
from typing import Union, Type, List, Optional

class ResNeSt(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 block: BottleneckResNetSplitAttention = BottleneckResNetSplitAttention,
                 layers: List[int] = [2, 2, 2, 2],
                 radix: int = 1,
                 groups: int = 1,
                 width_per_group: int = 64,
                 avg_down: bool = False,
                 avd: bool = False,
                 avd_first: bool = False,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 zero_init_residual: bool = False,
                 ) -> None:
        super().__init__()
        self.cardinality = groups
        self.radix = radix
        self.inplanes = 64
        self.dilation = 1
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.bottleneck_width = width_per_group
        self.conv1 = ConvBNAct(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, act="relu")
        self.maxpool = MaxPool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = AdaptiveAvgPool2D((1, 1))

        for m in self.modules():
            if isinstance(m, ConvBNAct):
                nn.init.kaiming_normal(m.conv.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ConvBNAct):
                    n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                    m.conv.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bn.weight.data.fill_(1)
                    m.bn.bias.data.zero_()

    def _make_layer(self, 
                    block: BottleneckResNetSplitAttention, 
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False,
                    is_first: bool = True) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBNAct(self.inplanes, planes * block.expansion, stride=stride, kernel_size=1, act="relu")

        layers = []
        layers.append(
                block(
                    self.inplanes, planes, stride, downsample, radix=self.radix, 
                    cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, 
                    avd = self.avd, avd_first=self.avd_first, dilation=previous_dilation, is_first=is_first
                )
            )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=self.dilation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return x
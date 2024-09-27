# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import MobileNetV3InvertedResidual, ConvBNAct, PyramidalPoolCBAM, Fusion4CBAM, VajraBottleneckBlock
from vajra.ops import make_divisible

class MobileNetV3(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 channels_list=[16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96],
                 kernel_sizes = [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
                 strides=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
                 dilations=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 expansion_ratios=[1, 6, 3.67, 4, 6, 6, 3, 3, 6, 6, 6],
                 se_bools = [True, False, False, True, True, True, True, True, True, True, True],
                 hs_activations = [False, False, False, True, True, True, True, True, True, True, True],
                 width_mul=1.0,
                 last_channel = 1024,
                 dropout=0.2,
                 num_classes=1000) -> None:
        super().__init__()
        block = MobileNetV3InvertedResidual
        input_channel = channels_list[0]
        input_channel = make_divisible(input_channel * width_mul, 8)
        self.input_channels = input_channel
        last_channel = make_divisible(last_channel * width_mul, 8)
        self.last_channel = last_channel
        self.conv1 = ConvBNAct(in_channels, input_channel, stride=2, kernel_size=3, act="hardswish")
        self.blocks = []
        self.from_list = [-1] * len(channels_list)

        for idx, (kernel_size, stride, dilation, channels, exp_ratio, use_hs, use_se) in enumerate(zip(kernel_sizes, strides, dilations, channels_list, expansion_ratios, hs_activations, se_bools)):
            output_channel = make_divisible(channels * width_mul, 8)
            self.blocks.append(block(input_channel, output_channel, stride, kernel_size, dilation, exp_ratio, use_se, use_hs))
            input_channel = output_channel

        last_conv_channel = 6 * output_channel

        self.blocks.append(ConvBNAct(output_channel, last_conv_channel, stride=1, kernel_size=1, act="hardswish"))
        self.features = nn.Sequential(*self.blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        conv1 = self.conv1(x)
        feats = self.features(conv1)
        avg_pool = self.pool(feats).flatten(1)
        out = self.classifier(avg_pool)
        return out
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import MobileNetV2InvertedResidual, ConvBNAct, Sanlayan, ChatushtayaSanlayan, VajraMerudandaBhag1
from vajra.ops import make_divisible

class MobileNetV2(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 channels_list=[16, 24, 32, 64, 96, 160, 320],
                 num_repeats=[1, 2, 3, 4, 3, 3, 1],
                 strides=[1, 2, 2, 2, 1, 2, 1],
                 expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
                 width_mul=1.0,
                 dropout=0.2,
                 num_classes=1000) -> None:
        super().__init__()
        block = MobileNetV2InvertedResidual
        input_channel = 32
        input_channel = make_divisible(input_channel * width_mul, 8)
        self.input_channels = input_channel
        last_channel = 1280
        self.last_channel = make_divisible(last_channel * width_mul, 8)
        self.conv1 = ConvBNAct(in_channels, input_channel, stride=2, act="relu6")
        self.blocks = []

        for idx, (num_blocks, stride, channels, exp_ratio) in enumerate(zip(num_repeats, strides, channels_list, expansion_ratios)):
            output_channel = make_divisible(channels * width_mul, 8)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                self.blocks.append(block(input_channel, output_channel, stride=s, expand_ratio=exp_ratio))
                input_channel = output_channel
        
        self.blocks.append(ConvBNAct(input_channel, self.last_channel, kernel_size=1, act="relu6"))
        self.features = nn.Sequential(*self.blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes)
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
        feats_out = self.features(conv1)
        avg_pool = self.pool(feats_out).flatten(1)
        out = self.classifier(avg_pool)
        return out
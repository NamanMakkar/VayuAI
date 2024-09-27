# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import DepthWiseSeparableConvBNAct, ConvBNAct

class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, act="relu", num_classes=1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=2, padding=1, act="relu"),
            DepthWiseSeparableConvBNAct(32, 64, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(64, 128, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(128, 128, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(128, 256, 2, 3, act=act),
            DepthWiseSeparableConvBNAct(256, 256, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(256, 512, 2, 3, act=act),
            DepthWiseSeparableConvBNAct(512, 512, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(512, 512, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(512, 512, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(512, 512, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(512, 512, 1, 3, act=act),
            DepthWiseSeparableConvBNAct(512, 1024, 2, 3, act=act),
            DepthWiseSeparableConvBNAct(1024, 1024, 1, 3, act=act),
        )

        self.pool = nn.AvgPool2d((7, 7))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        feats = self.features(x)
        pool = self.pool(feats).flatten(1)
        out = self.classifier(out)
        return out
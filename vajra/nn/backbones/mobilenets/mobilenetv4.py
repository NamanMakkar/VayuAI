# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import MobileNetV4InvertedResidual, ConvBNAct, PyramidalPoolCBAM, Fusion4CBAM, VajraBottleneckBlock
from vajra.ops import make_divisible

class MobileNetV4(nn.Module):
    def __init__(self, model_config, in_channels=3, num_classes=1000) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        self.from_list = [-1] * len(model_config)
        for block_type, *block_cfg in model_config:
            if block_type == "conv_bn":
                block = ConvBNAct
                out_channels, stride, kernel_size = block_cfg
                layers.append(block(channels, out_channels, stride, kernel_size, act="relu"))

            elif block_type == "inverted_residual":
                block = MobileNetV4InvertedResidual
                start_kernel_size, middle_kernel_size, stride, out_channels, expand_ratio = block_cfg
                layers.append(block(channels, out_channels, expand_ratio, start_kernel_size, middle_kernel_size, stride))
            
            else:
                raise NotImplementedError

            channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = 1280
        self.conv = ConvBNAct(channels, hidden_channels, 1, 1, act="relu")
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
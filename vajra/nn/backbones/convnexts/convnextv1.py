import torch
import torch.nn as nn
import torch.nn.functional as F
from vajra.utils import LOGGER
from vajra.nn.modules import Conv, ConvBNAct, ConvNeXtV1Block, LayerNorm, trunc_normal_, Linear
from vajra.nn.head import Detection, PoseDetection, Segmentation, OBBDetection, Panoptic

class ConvNeXtV1(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 channels_list=[96, 192, 384, 768],
                 num_blocks=[3, 3, 27, 3],
                 drop_path_rate = 0.,
                 head_init_scale=1.,
                 num_classes=1000,
                 layer_init_scale_value=1e-6,
                 ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.channels_list = channels_list
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            Conv(in_channels, channels_list[0], stride=4, kernel_size=4, bias=True),
            LayerNorm(channels_list[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(channels_list[i], eps=1e-6, data_format="channels_first"),
                    Conv(channels_list[i], channels_list[i+1], kernel_size=2, stride=2, bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV1Block(dim=channels_list[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_init_scale_value) for j in range(num_blocks[i])]
            )
            self.stages.append(stage)
            cur += num_blocks[i]

        #self.norm = LayerNorm(channels_list[-1], eps=1e-6)
        #self.head = Linear(channels_list[-1], num_classes)
        self.apply(self._init_weights)
        #self.head.linear.weight.data.mul_(head_init_scale)
        #self.head.linear.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, Conv):
            trunc_normal_(m.conv.weight, std=.02)
            nn.init.constant_(m.conv.bias, 0)

        if isinstance(m, Linear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)
        
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x #self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        return x
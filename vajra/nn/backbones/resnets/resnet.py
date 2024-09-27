# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import BasicBlock, BottleneckResNet, Linear, ConvBNAct, PyramidalPoolCBAM, Fusion4CBAM, VajraBottleneckBlock, MaxPool, AdaptiveAvgPool2D
from vajra.nn.head import Classification, Detection, PoseDetection, OBBDetection, Segementation
from vajra.utils import LOGGER
from vajra.ops import make_divisible
from typing import Union, Type, List, Optional

class ResNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 block: Type[Union[BasicBlock, BottleneckResNet]] = BasicBlock,
                 layers: List[int] = [2, 2, 2, 2],
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 zero_init_residual: bool = False,
                 ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvBNAct(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, act="relu")
        self.maxpool = MaxPool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
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
                if isinstance(m, BottleneckResNet) and m.conv3.bn.weight is not None:
                    nn.init.constant_(m.conv3.bn.weight, 0)

                elif isinstance(m, BasicBlock) and m.conv2.bn.weight is not None:
                    nn.init.constant_(m.conv2.bn.weight, 0)

    def _make_layer(self, 
                    block: Type[Union[BasicBlock, BottleneckResNet]], 
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
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
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation
                )
            )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
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

def build_resnet(in_channels,
                 num_classes,
                 size="18",
                 task="classify",
                 kpt_shape=None,
                 verbose=False):
    
    config_dict = {
        "18": {"block": BasicBlock,
               "layers": [2, 2, 2, 2],
               "groups": 1,
               "width_per_group": 64,
               },

        "34": {"block": BasicBlock,
               "layers": [3, 4, 6, 3],
               "groups": 1,
               "width_per_group": 64,
              },

        "50": {"block": BottleneckResNet,
               "layers": [3, 4, 6, 3],
               "groups": 1,
               "width_per_group": 64,
              },

        "101": {"block": BottleneckResNet,
                "layers": [3, 4, 23, 3],
                "groups": 1,
                "width_per_group": 64,
               },

        "152": {"block": BottleneckResNet,
                "layers": [3, 8, 36, 3],
                "groups": 1,
                "width_per_group": 64,
               },

        "50_32x4d": {"block": BottleneckResNet,
                     "layers": [3, 4, 6, 3],
                     "groups": 32,
                     "width_per_group": 4,
                    },

        "50_32x8d": {"block": BottleneckResNet,
                     "layers": [3, 4, 6, 3],
                     "groups": 32,
                     "width_per_group": 8,
                    },

        "101_64x4d": {"block": BottleneckResNet,
                      "layers": [3, 4, 23, 3],
                      "groups": 64,
                      "width_per_group": 4,
                     },

        "wide_50_2": {"block": BottleneckResNet,
                      "layers": [3, 4, 6, 3],
                      "groups": 1,
                      "width_per_group": 128,
                     },

        "wide_101_2": {"block": BottleneckResNet,
                       "layers": [3, 4, 23, 3],
                       "groups": 32,
                       "width_per_group": 4,
                       },
    }

    block = config_dict[size]["block"]
    layers = config_dict[size]["layers"]
    groups = config_dict[size]["groups"]
    width_per_groups = config_dict[size]["width_per_group"]

    if task == "classify":
        model = ResNet(in_channels, block, layers, groups, width_per_groups)
        head = Linear(512 * block.expansion, num_classes)
        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        layers = []
        layers.append(model)
        layers.append(head)

        resnet = nn.Sequential(*layers)
        np_model += np_head

        if verbose:
            LOGGER.info(f"Task: classify; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building ResNet ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")

            idx_counter = 0

            for i, (name, module) in enumerate(model.named_children()):
                if name in ["layer1", "layer2", "layer3", "layer4"]:
                    for idx, submodule in enumerate(module):
                        if isinstance(submodule, nn.Sequential):
                            for seq_idx, seq_module in enumerate(submodule):
                                if hasattr(seq_module, 'get_module_info'):
                                    np = sum(x.numel() for x in seq_module.parameters())
                                    module_info, args_info = seq_module.get_module_info()
                                    LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                                    idx_counter += 1
                        else:
                            if hasattr(submodule, 'get_module_info'):
                                np = sum(x.numel() for x in submodule.parameters())
                                module_info, args_info = submodule.get_module_info()
                                LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                                idx_counter += 1
                else:
                    if hasattr(module, 'get_module_info'):
                        np = sum(x.numel() for x in module.parameters())
                        module_info, args_info = module.get_module_info()
                        LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                        idx_counter += 1
        return resnet, head, np_model
    return
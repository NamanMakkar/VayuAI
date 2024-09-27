# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import MobileNetV4InvertedResidual, ConvBNAct, PyramidalPoolCBAM, Fusion4CBAM, VajraBottleneckBlock
from vajra.ops import make_divisible
from vajra.nn.backbones.mobilenets.mobilenet import MobileNetV1
from vajra.nn.backbones.mobilenets.mobilenetv2 import MobileNetV2
from vajra.nn.backbones.mobilenets.mobilenetv3 import MobileNetV3
from vajra.nn.backbones.mobilenets.mobilenetv4 import MobileNetV4

def build_mobilenet(in_channels, num_classes, size, version="v2", task="classify", verbose=False, kpt_shape=None, width_mul=1.0):
    v2_config = {
        "atto": {"width_mul": 0.1},
        "nano": {"width_mul": 0.25},
        "small": {"width_mul": 0.35},
        "medium": {"width_mul": 0.5},
        "large": {"width_mul": 0.75},
        "xlarge": {"width_mul": 1.0}
    }

    v3_config = {
        "small": {
            "channels_list": [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96],
            "kernel_sizes": [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
            "strides": [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
            "dilations": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "expansion_ratios": [1, 6, 3.67, 4, 6, 6, 3, 3, 6, 6, 6],
            "se_bools": [True, False, False, True, True, True, True, True, True, True, True],
            "hs_activations": [False, False, False, True, True, True, True, True, True, True, True],
            "last_channel": 1024
        },

        "large": {
            "channels_list": [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160],
            "kernel_sizes": [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5],
            "strides": [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
            "dilations": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "expansion_ratios": [1, 4, 3, 3, 3, 3, 6, 2.5, 2.3, 2.3, 6, 6, 6, 6, 6],
            "se_bools": [False, False, False, True, True, True, False, False, False, False, True, True, True, True, True],
            "hs_activations": [False, False, False, False, False, False, True, True, True, True, True, True, True, True, True],
            "last_channel": 1280,
        }
    }

    v4_config={
        "small":[
            ("conv_bn", 32, 2, 3),
            ("conv_bn", 32, 2, 3),
            ("conv_bn", 32, 1, 1),
            ("conv_bn", 96, 2, 3),
            ("conv_bn", 64, 1, 1),
            ("inverted_residual", 5, 5, 2, 96, 3.0),
            ("inverted_residual", 0, 3, 1, 96, 2.0),
            ("inverted_residual", 0, 3, 1, 96, 2.0),
            ("inverted_residual", 0, 3, 1, 96, 2.0),
            ("inverted_resudual", 0, 3, 1, 96, 2.0),
            ("inverted_residual", 3, 0, 1, 96, 4.0),
            ("inverted_residual", 3, 3, 2, 128, 6.0),
            ("inverted_residual", 5, 5, 1, 128, 4.0),
            ("inverted_residual", 0, 5, 1, 128, 4.0),
            ("inverted_residual", 0, 5, 1, 128, 3.0),
            ("inverted_residual", 0, 3, 1, 128, 4.0),
            ("inverted_residual", 0, 3, 1, 128, 4.0),
            ("conv_bn", 960, 1, 1),
        ],
        "medium": [
            ("conv_bn", 32, 2, 3),
            ("conv_bn", 128, 2, 3),
            ("conv_bn", 48, 1, 1),
            ("inverted_residual", 3, 5, 2, 80, 4.0),
            ("inverted_residual", 3, 3, 1, 80, 2.0),
            ("inverted_residual", 3, 5, 2, 160, 6.0),
            ("inverted_residual", 3, 3, 1, 160, 4.0),
            ("inverted_residual", 3, 3, 1, 160, 4.0),
            ("inverted_residual", 3, 3, 1, 160, 4.0),
            ("inverted_residual", 3, 3, 1, 160, 4.0),
            ("inverted_residual", 3, 0, 1, 160, 4.0),
            ("inverted_residual", 0, 0, 1, 160, 2.0),
            ("inverted_residual", 3, 0, 1, 160, 4.0),

            ('inverted_residual', 5, 5, 2, 256, 6.0),
            ('inverted_residual', 5, 5, 1, 256, 4.0),
            ('inverted_residual', 3, 5, 1, 256, 4.0),
            ('inverted_residual', 3, 5, 1, 256, 4.0),
            ('inverted_residual', 0, 0, 1, 256, 4.0),
            ('inverted_residual', 3, 0, 1, 256, 4.0),
            ('inverted_residual', 3, 5, 1, 256, 2.0),
            ('inverted_residual', 5, 5, 1, 256, 4.0),
            ('inverted_residual', 0, 0, 1, 256, 4.0),
            ('inverted_residual', 0, 0, 1, 256, 4.0),
            ('inverted_residual', 5, 0, 1, 256, 2.0),
        ],

        "large": [
            ('conv_bn', 24, 2, 3),
            ('conv_bn', 96, 2, 3),
            ('conv_bn', 48, 1, 1),
            ('inverted_residual', 3, 5, 2, 96, 4.0),
            ('inverted_residual', 3, 3, 1, 96, 4.0),
            ('inverted_residual', 3, 5, 2, 192, 4.0),
            ('inverted_residual', 3, 3, 1, 192, 4.0),
            ('inverted_residual', 3, 3, 1, 192, 4.0),
            ('inverted_residual', 3, 3, 1, 192, 4.0),
            ('inverted_residual', 3, 5, 1, 192, 4.0),
            ('inverted_residual', 5, 3, 1, 192, 4.0),
            ('inverted_residual', 5, 3, 1, 192, 4.0),
            ('inverted_residual', 5, 3, 1, 192, 4.0),
            ('inverted_residual', 5, 3, 1, 192, 4.0),
            ('inverted_residual', 5, 3, 1, 192, 4.0),
            ('inverted_residual', 3, 0, 1, 192, 4.0),
            ('inverted_residual', 5, 5, 2, 512, 4.0),
            ('inverted_residual', 5, 5, 1, 512, 4.0),
            ('inverted_residual', 5, 5, 1, 512, 4.0),
            ('inverted_residual', 5, 5, 1, 512, 4.0),
            ('inverted_residual', 5, 0, 1, 512, 4.0),
            ('inverted_residual', 5, 3, 1, 512, 4.0),
            ('inverted_residual', 5, 0, 1, 512, 4.0),
            ('inverted_residual', 5, 0, 1, 512, 4.0),
            ('inverted_residual', 5, 3, 1, 512, 4.0),
            ('inverted_residual', 5, 5, 1, 512, 4.0),
            ('inverted_residual', 5, 0, 1, 512, 4.0),
            ('inverted_residual', 5, 0, 1, 512, 4.0),
            ('inverted_residual', 5, 0, 1, 512, 4.0),
            ("conv_bn", 960, 1, 1),
        ],
    }

    v2_width_mul = v2_config[size]["width_mul"]
    v3_channels = v3_config[size]["channels_list"]
    v3_kernel_sizes = v3_config[size]["kernel_sizes"]
    v3_strides = v3_config[size]["strides"]
    v3_dilations = v3_config[size]["dilations"]
    v3_expansion_ratios = v3_config[size]["expansion_ratios"]
    v3_se_bools = v3_config[size]["se_bools"]
    v3_hs_activations = v3_config[size]["hs_activations"]
    v3_last_channels = v3_config[size]["last_channel"]

    v4_block_config = v4_config[size]

    if task == "classify":
        if version == "v1":
            model = MobileNetV1(in_channels, num_classes=num_classes)
        elif version == "v2":
            model = MobileNetV2(in_channels, width_mul=v2_width_mul, num_classes=num_classes)
        elif version == "v3":
            model = MobileNetV3(in_channels, v3_channels, v3_kernel_sizes, v3_strides, v3_dilations, v3_expansion_ratios, v3_se_bools, v3_hs_activations, width_mul, v3_last_channels, num_classes=num_classes)
        elif version == "v4":
            model = MobileNetV4(v4_block_config, in_channels, num_classes)

        np_model = sum(x.numel() for x in model.parameters())
        
    return
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import MBConvEffNet, ConvBNAct, Sanlayan, ChatushtayaSanlayan, VajraMerudandaBhag1
from vajra.nn.head import Detection, Segmentation, OBBDetection, PoseDetection, Panoptic
from vajra.utils import LOGGER
from vajra.ops import make_divisible

class EfficientNetV1(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[16, 24, 40, 80, 112, 192, 320],
                 num_repeats=[1, 2, 2, 3, 3, 4, 1],
                 strides=[1, 2, 2, 2, 1, 2, 1],
                 expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
                 kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 width_mul=1.0,
                 depth_mul=1.0) -> None:
        super().__init__()
        block_channels = make_divisible(32 * width_mul, 8)
        self.conv1 = ConvBNAct(in_channels, block_channels, stride=2, kernel_size=3)
        self.blocks = []
        for idx, (channels, num_blocks, stride, kernel, expansion_ratio) in enumerate(zip(channels_list, num_repeats, strides, kernel_sizes, expansion_ratios)):
            channels = make_divisible(channels * width_mul, 8)
            num_blocks = math.ceil(num_blocks * depth_mul)
            for i in range(num_blocks):
                    self.blocks.append(MBConvEffNet(in_c=block_channels, out_c=channels, stride=stride if i == 0 else 1, expansion_ratio=expansion_ratio, kernel_size=kernel))
                    block_channels = channels

        self.features = nn.Sequential(*self.blocks)

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.features(conv1)
        return out

class VajraEffNetV1(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[16, 24, 40, 80, 112, 192, 320],
                 neck_channels_list = [160, 160, 160, 160, 160, 160, 160, 160],
                 num_repeats=[1, 2, 2, 3, 3, 4, 1],
                 num_neck_repeats=[1, 1, 1, 1],
                 strides=[1, 2, 2, 2, 1, 2, 1],
                 expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
                 kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 width_mul=1.0,
                 depth_mul=1.0) -> None:
        super().__init__()
        block_channels = make_divisible(32 * width_mul, 8)
        self.channels_list = [make_divisible(ch * width_mul, 8) for ch in channels_list]
        self.num_repeats = [math.ceil(n * depth_mul) for n in num_repeats]
        self.neck_channels_list = [make_divisible(neck_ch * width_mul, 8) for neck_ch in neck_channels_list]
        self.num_neck_repeats = [math.ceil(n * depth_mul) for n in num_neck_repeats]
        sum_backbone_repeats = sum(self.num_repeats)
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [-1-self.num_repeats[-1]-self.num_repeats[-2]-self.num_repeats[-3], -1-self.num_repeats[-1]-self.num_repeats[-2], -1-self.num_repeats[-1], -1], [-2-self.num_repeats[-1]-self.num_repeats[-2]-self.num_repeats[-3], -2-self.num_repeats[-1]-self.num_repeats[-2], -2-self.num_repeats[-1],-1], -1, [-4-self.num_repeats[-1]-self.num_repeats[-2]-self.num_repeats[-3], -4-self.num_repeats[-1], -4-self.num_repeats[-1]-self.num_repeats[-2],-1], -1, [-5, -3, -1], -1, [-5, -3, -1], -1, [sum_backbone_repeats + 5, sum_backbone_repeats + 7, sum_backbone_repeats + 9]]

        self.conv1 = ConvBNAct(in_channels, block_channels, stride=2, kernel_size=3)
        self.blocks1 = nn.ModuleList(MBConvEffNet(in_c=block_channels if i == 0 else self.channels_list[0], out_c=self.channels_list[0], stride=strides[0] if i == 0 else 1, expansion_ratio=expansion_ratios[0], kernel_size=kernel_sizes[0]) for i in range(self.num_repeats[0]))
        self.blocks2 = nn.ModuleList(MBConvEffNet(in_c=self.channels_list[0] if i==0 else self.channels_list[1], out_c=self.channels_list[1], stride=strides[1] if i==0 else 1, expansion_ratio=expansion_ratios[1], kernel_size=kernel_sizes[1]) for i in range(self.num_repeats[1]))
        self.blocks3 = nn.ModuleList(MBConvEffNet(in_c=self.channels_list[1] if i==0 else self.channels_list[2], out_c=self.channels_list[2], stride=strides[2] if i==0 else 1, expansion_ratio=expansion_ratios[2], kernel_size=kernel_sizes[2]) for i in range(self.num_repeats[2]))
        self.blocks4 = nn.ModuleList(MBConvEffNet(in_c=self.channels_list[2] if i==0 else self.channels_list[3], out_c=self.channels_list[3], stride=strides[3] if i==0 else 1, expansion_ratio=expansion_ratios[3], kernel_size=kernel_sizes[3]) for i in range(self.num_repeats[3]))
        self.blocks5 = nn.ModuleList(MBConvEffNet(in_c=self.channels_list[3] if i==0 else self.channels_list[4], out_c=self.channels_list[4], stride=strides[4] if i==0 else 1, expansion_ratio=expansion_ratios[4], kernel_size=kernel_sizes[4]) for i in range(self.num_repeats[4]))
        self.blocks6 = nn.ModuleList(MBConvEffNet(in_c=self.channels_list[4] if i==0 else self.channels_list[5], out_c=self.channels_list[5], stride=strides[5] if i==0 else 1, expansion_ratio=expansion_ratios[5], kernel_size=kernel_sizes[5]) for i in range(self.num_repeats[5]))
        self.blocks7 = nn.ModuleList(MBConvEffNet(in_c=self.channels_list[5] if i==0 else self.channels_list[6], out_c=self.channels_list[6], stride=strides[6] if i==0 else 1, expansion_ratio=expansion_ratios[6], kernel_size=kernel_sizes[6]) for i in range(self.num_repeats[6]))
        self.pyramidal_pool_cbam = Sanlayan(in_c=self.channels_list[3:], out_c=self.channels_list[6], stride=2)
        
        self.fusion4cbam = ChatushtayaSanlayan(in_c=self.channels_list[3:], out_c=self.neck_channels_list[0])
        self.vajra_neck1 = VajraMerudandaBhag1(in_c=self.neck_channels_list[0], out_c=self.neck_channels_list[1], num_blocks=self.num_neck_repeats[0], shortcut=False, kernel_size=1, bottleneck_dwcib=False)

        self.fusion4cbam_1 = ChatushtayaSanlayan(in_c=[self.channels_list[4], self.channels_list[5], self.channels_list[3], self.neck_channels_list[1]], out_c=self.neck_channels_list[2])
        self.vajra_neck2 = VajraMerudandaBhag1(in_c=self.neck_channels_list[2], out_c=self.neck_channels_list[3], num_blocks=self.num_neck_repeats[1], shortcut=False, kernel_size=1, bottleneck_dwcib=False)

        self.pyramidal_pool_neck = Sanlayan(in_c=[self.channels_list[6], self.neck_channels_list[1], self.neck_channels_list[3]], out_c=self.neck_channels_list[4], stride=2)
        self.vajra_neck3 = VajraMerudandaBhag1(in_c=self.neck_channels_list[4], out_c=self.neck_channels_list[5], num_blocks=self.num_neck_repeats[2], shortcut=False, kernel_size=1, bottleneck_dwcib=False)

        self.pyramidal_pool_neck_1 = Sanlayan(in_c=[self.neck_channels_list[1], self.neck_channels_list[3], self.neck_channels_list[5]], out_c=self.neck_channels_list[6], stride=2)
        self.vajra_neck4 = VajraMerudandaBhag1(in_c=self.neck_channels_list[6], out_c=self.neck_channels_list[7], num_blocks=self.num_neck_repeats[3], shortcut=False, kernel_size=1, bottleneck_dwcib=False)

    def forward(self, x):
        x = self.conv1(x)
        for i,block in enumerate(self.blocks1):
            if i == 0:
                blocks1 = block(x)
            else:
                blocks1 = block(blocks1)

        for i, block in enumerate(self.blocks2):
            if i == 0:
                blocks2 = block(blocks1)
            else:
                blocks2 = block(blocks2)
        
        for i, block in enumerate(self.blocks3):
            if i == 0:
                blocks3 = block(blocks2)
            else:
                blocks3 = block(blocks3)
        
        for i, block in enumerate(self.blocks4):
            if i == 0:
                blocks4 = block(blocks3)
            else:
                blocks4 = block(blocks4)

        for i, block in enumerate(self.blocks5):
            if i == 0:
                blocks5 = block(blocks4)
            else:
                blocks5 = block(blocks5)

        for i, block in enumerate(self.blocks6):
            if i == 0:
                blocks6 = block(blocks5)
            else:
                blocks6 = block(blocks6)

        for i, block in enumerate(self.blocks7):
            if i == 0:
                blocks7 = block(blocks6)
            else:
                blocks7 = block(blocks7)

        pyramidal_pool_cbam = self.pyramidal_pool_cbam([blocks4, blocks5, blocks6, blocks7])

        fusion4 = self.fusion4cbam([blocks4, blocks5, blocks6, pyramidal_pool_cbam])
        vajra_neck1 = self.vajra_neck1(fusion4)

        fusion4_1 = self.fusion4cbam_1([blocks4, blocks6, blocks5, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_1)

        pyramidal_pool_cbam_neck1 = self.pyramidal_pool_neck([pyramidal_pool_cbam, vajra_neck1, vajra_neck2])
        vajra_neck3 = self.vajra_neck3(pyramidal_pool_cbam_neck1)

        pyramidal_pool_cbam_neck2 = self.pyramidal_pool_neck_1([vajra_neck1, vajra_neck2, vajra_neck3])
        vajra_neck4 = self.vajra_neck4(pyramidal_pool_cbam_neck2)
        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class EffnetV1CLSHead(nn.Module):
    def __init__(self, in_c, out_c, hidden_c=2048, dropout=0.2) -> None:
        super().__init__()
        self.hidden_c = hidden_c
        self.in_c = in_c
        self.out_c = out_c
        self.dropout = dropout
        self.conv = ConvBNAct(in_c, hidden_c, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_c, self.out_c),
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear((self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)

    def get_module_info(self):
        return f"EffnetV1CLSHead", f"[{self.in_c}, {self.out_c}, {self.hidden_c}, {self.dropout}]"
    
def build_effnetv1(in_channels,
                   num_classes,
                   size="b0",
                   task="detect",
                   verbose=False,
                   kpt_shape=None,):
    alpha = 1.2
    beta = 1.1
    config_dict = {
        "b0": {"phi": 0, "dropout": 0.2,
               "width_mul": 1.0,
               "depth_mul": 1.0},

        "b1": {"phi": 0.5, "dropout": 0.2,
               "width_mul": 1.0,
               "depth_mul": 1.1},

        "b2": {"phi": 1, "dropout": 0.3,
               "width_mul": 1.1,
               "depth_mul": 1.2},

        "b3": {"phi": 2, "dropout": 0.3,
               "width_mul": 1.2,
               "depth_mul": 1.4},

        "b4": {"phi": 3, "dropout": 0.4,
               "width_mul": 1.4,
               "depth_mul": 1.8},

        "b5": {"phi": 4, "dropout": 0.4,
               "width_mul": 1.6,
               "depth_mul": 2.2},

        "b6": {"phi": 5, "dropout": 0.5,
               "width_mul": 1.8,
               "depth_mul": 2.6},

        "b7": {"phi": 6, "dropout": 0.5,
               "width_mul": 2.0,
               "depth_mul": 3.1}
    }

    phi = config_dict[size]["phi"]
    dropout = config_dict[size]["dropout"]
    depth_mul = config_dict[size]["depth_mul"]
    width_mul = config_dict[size]["width_mul"]

    if task == "classify":
        model = EfficientNetV1(in_channels, width_mul=width_mul, depth_mul=depth_mul)
        head = EffnetV1CLSHead(in_c=make_divisible(320 * width_mul, 8), out_c=num_classes, hidden_c=math.ceil(1280 * width_mul), dropout=dropout)

        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        layers = []
        layers.append(model)
        layers.append(head)

        effnetv1 = nn.Sequential(*layers)
        np_model += np_head

        if verbose:
            LOGGER.info(f"Task: classify; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building EfficientNetV1 ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            conv1 = model.conv1
            np_conv1 = sum(x.numel() for x in conv1.parameters())
            md_info_conv1, arg_info_conv1 = conv1.get_module_info()
            LOGGER.info(f"{0:>3}.{str(-1):>20}{np_conv1:>10}  {md_info_conv1:<45}{arg_info_conv1:<30}")


            for i, block in enumerate(model.blocks):
                np = sum(x.numel() for x in block.parameters())
                md_info, args_info = block.get_module_info()
                LOGGER.info(f"{i+1:>3}.{str(-1):>20}{np:>10}  {md_info:<45}{args_info:<30}")

            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{i+1:>3}.{str(-1):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nBackbone Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"EfficientNetV1-{size}; Task: classify; Total Parameters: {np_model}\n\n")
        
        return effnetv1, layers, np_model

    else:
        head_channels = [160, 160, 160]
        stride = torch.tensor([8., 16., 32.])
        head_channels = [make_divisible(head_ch * width_mul, 8) for head_ch in head_channels]
        model = VajraEffNetV1(in_channels=in_channels, width_mul=width_mul, depth_mul=depth_mul)
        if task == "detect":
            head = Detection(num_classes, head_channels)
        elif task == "segment":
            head = Segmentation(num_classes, in_channels=head_channels)
        elif task == "pose":
            head = PoseDetection(num_classes, in_channels=head_channels, keypoint_shape=kpt_shape if any(kpt_shape) else (17, 3))
        elif task == "obb":
            head = OBBDetection(num_classes, in_channels=head_channels)

        layers = []
        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        if verbose:
            LOGGER.info(f"Task: {task}; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building VajraEffNetV1 ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            idx_counter = 0
            for i, (name, module) in enumerate(model.named_children()):
                if isinstance(module, nn.ModuleList):
                    for idx, block in enumerate(module):
                        if isinstance(block, MBConvEffNet):
                            np = sum(x.numel() for x in block.parameters())
                            md_info, arg_info = block.get_module_info()
                            LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {md_info:<45}{arg_info:<30}")
                            idx_counter += 1
                else:
                    np = sum(x.numel() for x in module.parameters())
                    md_info, arg_info = module.get_module_info()
                    LOGGER.info(f"{idx_counter:>3}.{str(model.from_list[i]):>20}{np:>10}  {md_info:<45}{arg_info:<30}")
                    idx_counter += 1

            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{i+1:>3}.{str(model.from_list[-1]):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nBackbone Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"VajraEffNetV1 {size}; Task: {task}; Total Parameters: {np_model}\n\n")

        np_model += np_head
        head.stride = stride
        head.bias_init()
        layers.append(model)
        layers.append(head)
        vajra_effnetv1 = nn.Sequential(*layers)

        return vajra_effnetv1, stride, layers, np_model
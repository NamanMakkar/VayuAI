# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from vajra.nn.modules import MSBlock, FusedMBConvEffNet, MBConvEffNet, ConvBNAct, Sanlayan, ChatushtayaSanlayan, VajraMerudandaBhag1
from vajra.nn.head import Classification, Detection, PoseDetection, OBBDetection, Segementation
from vajra.utils import LOGGER
from vajra.ops import make_divisible

class MENeSt(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 channels_list=[24, 48, 64, 128, 160, 256], 
                 num_repeats=[2, 4, 4, 6, 9, 15], 
                 strides=[1, 2, 2, 2, 1, 2],
                 use_se_bools = [False, False, False, True, True, True],
                 expansion_ratios=[1, 4, 4, 4, 6, 6],
                 width_mul=1.) -> None:
        super().__init__()
        block_channels = make_divisible(24 * width_mul, 8)
        self.conv1 = ConvBNAct(in_channels, block_channels, stride=2, kernel_size=3)
        self.blocks = []
        for idx, (channels, num_blocks, stride, use_se, expansion_ratio) in enumerate(zip(channels_list, num_repeats, strides, use_se_bools, expansion_ratios)):
            channels = make_divisible(channels * width_mul, 8)
            for i in range(num_blocks):
                self.blocks.append(MSBlock(in_c=block_channels, out_c=channels, stride=stride if i == 0 else 1, use_se=use_se, expand_ratio=expansion_ratio))
                block_channels = channels

        self.features = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.features(conv1)
        return out

class VajraMENeSt(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[24, 48, 64, 128, 160, 256], 
                 num_repeats=[2, 4, 4, 6, 9, 15], 
                 strides=[1, 2, 2, 2, 1, 2],
                 use_se_bools = [False, False, False, True, True, True],
                 expansion_ratios=[1, 4, 4, 4, 6, 6],
                 width_mul=1.,
                 neck_channels_list=[160, 160, 160, 160, 160, 160, 160, 160],
                 num_neck_repeats=[1, 1, 1, 1],
                 ) -> None:
        super().__init__()
        block_channels = make_divisible(24 * width_mul, 8)
        self.channels_list = [make_divisible(ch * width_mul, 8) for ch in channels_list]
        self.from_list = [-1] * 8
        self.is_small = len(self.channels_list) == 6
        sum_backbone_repeats = sum(num_repeats)
        if self.is_small:
            self.from_list.extend([[-1-num_repeats[-1]-num_repeats[-2]-num_repeats[-3], -1-num_repeats[-1]-num_repeats[-2], -1-num_repeats[-1], -1], [-2-num_repeats[-1]-num_repeats[-2], -2-num_repeats[-1], -2, -1], -1, [-4-num_repeats[-1]-num_repeats[-2], -4-num_repeats[-1], -4, -3], -1, [-5, -3, -1], -1, [-5, -3, -1], -1, [sum_backbone_repeats + 5, sum_backbone_repeats + 7, sum_backbone_repeats + 9]])
        else:
            self.from_list.extend([[-1-num_repeats[-1]-num_repeats[-2]-num_repeats[-3], -1-num_repeats[-1]-num_repeats[-2], -1-num_repeats[-1], -1], [-2-num_repeats[-1]-num_repeats[-2]-num_repeats[-3], -2-num_repeats[-1]-num_repeats[-2], -2-num_repeats[-1],-1], -1, [-4-num_repeats[-1]-num_repeats[-2]-num_repeats[-3], -4-num_repeats[-1], -4-num_repeats[-1]-num_repeats[-2],-1], -1, [-5, -3, -1], -1, [-5, -3, -1], -1, [sum_backbone_repeats + 5, sum_backbone_repeats + 7, sum_backbone_repeats + 9]])
        self.conv1 = ConvBNAct(in_channels, block_channels, stride=2, kernel_size=3)
        self.blocks1 = nn.ModuleList(MSBlock(in_c=block_channels if i == 0 else self.channels_list[0], out_c=self.channels_list[0], stride=strides[0] if i == 0 else 1, use_se=use_se_bools[0], expand_ratio=expansion_ratios[0]) for i in range(num_repeats[0]))
        self.blocks2 = nn.ModuleList(MSBlock(in_c=self.channels_list[0] if i == 0 else self.channels_list[1], out_c=self.channels_list[1], stride=strides[1] if i == 0 else 1, use_se=use_se_bools[1], expand_ratio=expansion_ratios[1]) for i in range(num_repeats[1]))
        self.blocks3 = nn.ModuleList(MSBlock(in_c=self.channels_list[1] if i == 0 else self.channels_list[2], out_c=self.channels_list[2], stride=strides[2] if i == 0 else 1, use_se=use_se_bools[2], expand_ratio=expansion_ratios[2]) for i in range(num_repeats[2]))
        self.blocks4 = nn.ModuleList(MSBlock(in_c=self.channels_list[2] if i == 0 else self.channels_list[3], out_c=self.channels_list[3], stride=strides[3] if i == 0 else 1, use_se=use_se_bools[3], expand_ratio=expansion_ratios[3]) for i in range(num_repeats[3]))
        self.blocks5 = nn.ModuleList(MSBlock(in_c=self.channels_list[3] if i == 0 else self.channels_list[4], out_c=self.channels_list[4], stride=strides[4] if i == 0 else 1, use_se=use_se_bools[4], expand_ratio=expansion_ratios[4]) for i in range(num_repeats[4]))
        self.blocks6 = nn.ModuleList(MSBlock(in_c=self.channels_list[4] if i == 0 else self.channels_list[5], out_c=self.channels_list[5], stride=strides[5] if i == 0 else 1, use_se=use_se_bools[5], expand_ratio=expansion_ratios[5]) for i in range(num_repeats[5]))
        self.blocks7 = nn.Identity() if self.is_small else nn.ModuleList(MSBlock(in_c=self.channels_list[5] if i == 0 else self.channels_list[6], out_c=self.channels_list[6], stride=strides[6] if i == 0 else 1, use_se=use_se_bools[6], expand_ratio=expansion_ratios[6]) for i in range(num_repeats[6]))
        self.pyramidal_pool_cbam = Sanlayan(in_c=self.channels_list[2:], out_c=self.channels_list[-1], stride=2) if self.is_small else Sanlayan(in_c=self.channels_list[-4:], out_c=self.channels_list[-1], stride=2)

        self.fusion4cbam = ChatushtayaSanlayan(in_c=self.channels_list[-4:], out_c=neck_channels_list[0]) if not self.is_small else ChatushtayaSanlayan(in_c=[self.channels_list[-3], self.channels_list[-2], self.channels_list[-1], self.channels_list[-1]], out_c=neck_channels_list[0])
        self.vajra_neck1 = VajraMerudandaBhag1(neck_channels_list[0], neck_channels_list[1], num_neck_repeats[0], False, 1, False)

        self.fusion4cbam_1 = ChatushtayaSanlayan(in_c=[self.channels_list[3], self.channels_list[4], self.channels_list[5], neck_channels_list[1]], out_c=neck_channels_list[2])
        self.vajra_neck2 = VajraMerudandaBhag1(neck_channels_list[2], neck_channels_list[3], num_neck_repeats[1], False, 1, False)

        self.pyramidal_pool_neck = Sanlayan(in_c=[channels_list[-1], neck_channels_list[1], neck_channels_list[3]], out_c=neck_channels_list[4], stride=2)
        self.vajra_neck3 = VajraMerudandaBhag1(neck_channels_list[4], neck_channels_list[5], num_neck_repeats[2], False, 1, False)

        self.pyramidal_pool_neck_1 = Sanlayan(in_c=[neck_channels_list[1], neck_channels_list[3], neck_channels_list[5]], out_c=neck_channels_list[6], stride=2)
        self.vajra_neck4 = VajraMerudandaBhag1(neck_channels_list[6], neck_channels_list[7], num_neck_repeats[3], False, 1, False)
    
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

        if not self.is_small:
            for i, block in enumerate(self.blocks7):
                if i == 0:
                    blocks7 = block(blocks6)
                else:
                    blocks7 = block(blocks7)

        pyramidal_pool_cbam = self.pyramidal_pool_cbam([blocks4, blocks5, blocks6, blocks7]) if not self.is_small else self.pyramidal_pool_cbam([blocks3, blocks4, blocks5, blocks6])

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

def build_me_nest(in_channels,
                  num_classes,
                  size="small",
                  task="classify",
                  kpt_shape=None,
                  verbose=False):
    config_dict = {
        "small": {"channels_list": [24, 48, 64, 128, 160, 256], 
                  "num_repeats": [2, 4, 4, 6, 9, 15],
                  "strides": [1, 2, 2, 2, 1, 2],
                  "use_se_bools": [False, False, False, True, True, True],
                  "expansion_ratios": [1, 4, 4, 4, 6, 6],
                  "width_mul": 1.0
                },

        "medium": {"channels_list": [24, 48, 80, 160, 176, 304, 512],
                   "num_repeats": [3, 5, 5, 7, 14, 18, 5], 
                   "strides": [1, 2, 2, 2, 1, 2, 1],
                   "use_se_bools": [False, False, False, True, True, True, True],
                   "expansion_ratios": [1, 4, 4, 4, 6, 6, 6],
                   "width_mul": 1.0
                },

        "large": {"channels_list": [32, 64, 96, 192, 224, 384, 640],
                  "num_repeats": [4, 7, 7, 10, 19, 25, 7],
                  "strides": [1, 2, 2, 2, 1, 2, 1],
                  "use_se_bools": [False, False, False, True, True, True, True],
                  "expansion_ratios": [1, 4, 4, 4, 6, 6, 6],
                  "width_mul": 1.0
                },

        "xlarge": {"channels_list": [32, 64, 96, 192, 256, 512, 640],
                   "num_repeats": [4, 8, 8, 16, 24, 32, 8],
                   "strides": [1, 2, 2, 2, 1, 2, 1],
                   "use_se_bools": [False, False, False, True, True, True, True],
                   "expansion_ratios": [1, 4, 4, 4, 6, 6, 6],
                   "width_mul": 1.0
                },
    }

    channels_list = config_dict[size]["channels_list"]
    num_repeats = config_dict[size]["num_repeats"]
    strides = config_dict[size]["strides"]
    use_se_bools = config_dict[size]["use_se_bools"]
    expansion_ratios = config_dict[size]["expansion_ratios"]
    width_mul = config_dict[size]["width_mul"]

    if task == "classify":
        model = MENeSt(in_channels=in_channels, channels_list=channels_list, num_repeats=num_repeats, strides=strides, use_se_bools=use_se_bools, expansion_ratios=expansion_ratios, width_mul=width_mul)
        head = Classification(in_c = make_divisible(channels_list[-1] * width_mul, 8), out_c = num_classes, hidden_c = make_divisible(1792 * width_mul, 8) if width_mul > 1. else 1792)
        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        layers = []
        layers.append(model)
        layers.append(head)

        menest = nn.Sequential(*layers)
        np_model += np_head

        if verbose:
            LOGGER.info(f"Task: classify; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building ME-NeSt ...\n\n")
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
            LOGGER.info(f"ME-NeSt-{size}; Task: classify; Total Parameters: {np_model}\n\n")

        return menest, layers, np_model

    else:
        config_dict_neck = {
                   "small": [0.33, 0.5, 1024], 
                   "medium": [0.67, 0.75, 768], 
                   "large": [1.0, 1.0, 512], 
                   "xlarge": [1.0, 1.25, 512],
                }
        width_mul_neck = config_dict_neck[size][1]
        depth_mul_neck = config_dict_neck[size][0]
        head_channels = [256, 256, 256]
        stride = torch.tensor([8., 16., 32.])

        head_channels = [make_divisible(head_ch * width_mul_neck, 8) for head_ch in head_channels]
        neck_channels_list = [256, 256, 256, 256, 256, 256, 256, 256]
        neck_channels_list = [make_divisible(neck_ch * width_mul_neck, 8) for neck_ch in neck_channels_list]
        num_neck_repeats = [3, 3, 3, 3]
        num_neck_repeats = [math.ceil(depth_mul_neck * num) for num in num_neck_repeats]
        model = VajraMENeSt(in_channels, channels_list, num_repeats, strides, use_se_bools, expansion_ratios, neck_channels_list=neck_channels_list, num_neck_repeats=num_neck_repeats)
        
        if task == "detect":
            head = Detection(num_classes, head_channels)
        elif task == "segment":
            head = Segementation(num_classes, in_channels=head_channels)
        elif task == "pose":
            head = PoseDetection(num_classes, in_channels=head_channels, keypoint_shape=kpt_shape if any(kpt_shape) else (17, 3))
        elif task == "obb":
            head = OBBDetection(num_classes, in_channels=head_channels)

        layers = []
        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        if verbose:
            LOGGER.info(f"Task: {task}; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building VajraMENeSt ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            idx_counter = 0
            #idx = 0
            for i, (name, module) in enumerate(model.named_children()):
                if isinstance(module, nn.ModuleList):
                    for idx, block in enumerate(module):
                        if isinstance(block, MSBlock):
                            np = sum(x.numel() for x in block.parameters())
                            md_info, arg_info = block.get_module_info()
                            LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {md_info:<45}{arg_info:<30}")
                            idx_counter += 1
                else:
                    if not isinstance(module, nn.Identity):
                        np = sum(x.numel() for x in module.parameters())
                        md_info, arg_info = module.get_module_info()
                        LOGGER.info(f"{idx_counter:>3}.{str(model.from_list[i]):>20}{np:>10}  {md_info:<45}{arg_info:<30}")
                        idx_counter += 1

            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{i+1:>3}.{str(model.from_list[-1]):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nBackbone Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"VajraMENeSt {size}; Task: {task}; Total Parameters: {np_model}\n\n")

        np_model += np_head
        head.stride = stride
        head.bias_init()
        layers.append(model)
        layers.append(head)
        vajra_menest = nn.Sequential(*layers)
        return vajra_menest, stride, layers, np_model
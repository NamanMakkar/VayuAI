import torch
import torch.nn as nn
import torch.nn.functional as F
from vajra.utils import LOGGER
from vajra.nn.modules import Conv, ConvBNAct, ConvNeXtV2Block, LayerNorm, trunc_normal_, Linear
from vajra.nn.head import Detection, PoseDetection, Segementation, OBBDetection, Panoptic

class ConvNeXtV2(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 channels_list=[96, 192, 384, 768],
                 num_blocks=[3, 3, 27, 3],
                 drop_path_rate = 0.,
                 head_init_scale=1.,
                 num_classes=1000,
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
                *[ConvNeXtV2Block(dim=channels_list[i], drop_path=dp_rates[cur + j]) for j in range(num_blocks[i])]
            )
            self.stages.append(stage)
            cur += num_blocks[i]

        self.apply(self._init_weights)

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

def build_convnextv2(in_channels,
                     num_classes,
                     size="small",
                     task="classify",
                     kpt_shape=None,
                     verbose=False
                     ):

    config_dict = {
        "tiny": {"channels_list": [96, 192, 384, 768],
                 "num_blocks": [3, 3, 9, 3],
                },
        
        "small": {"channels_list": [96, 192, 384, 768],
                  "num_blocks": [3, 3, 27, 3],
                },

        "base": {"channels_list": [128, 256, 512, 1024],
                 "num_blocks": [3, 3, 27, 3]
                },

        "large": {"channels_list": [192, 384, 768, 1536],
                  "num_blocks": [3, 3, 27, 3],
                },

        "xlarge": {"channels_list": [256, 512, 1024, 2048],
                   "num_blocks": [3, 3, 27, 3],
                },
    }

    channels_list = config_dict[size]["channels_list"]
    num_blocks = config_dict[size]["num_blocks"]

    if task == "classify":
        model = ConvNeXtV2(in_channels, channels_list, num_blocks, num_classes=num_classes)
        head = model.head
        np_model = sum(x.numel() for x in model.parameters())

        if verbose:            
            LOGGER.info(f"Task: classify; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building ConvNeXtV2 ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            idx_counter = 0
            for i, (name, module) in enumerate(model.named_children()):
                if name in ["downsample_layers", "stages"]:
                    for idx, submodule in enumerate(module):
                        np = sum(x.numel() for x in submodule.parameters())
                        module_info, args_info = submodule.get_module_info()
                        LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                        idx_counter += 1

                else:
                    if hasattr(module, 'get_module_info'):
                        np = sum(x.numel() for x in module.parameters())
                        module_info, args_info = module.get_module_info()
                        LOGGER.info(f"{idx_counter:>3}.{str(model.from_list[i]):>20}{np:>10}  {module_info:<45}{args_info:<30}")
                        idx_counter += 1
            
            

            LOGGER.info(f"\nModel Parameters: {np_model}\n\n")
            LOGGER.info(f"ConvNeXtV2 {size} Task: classify; Total Parameters: {np_model}\n\n")

        return model, head, np_model
    return
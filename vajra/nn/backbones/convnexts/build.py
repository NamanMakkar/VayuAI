import torch
import torch.nn as nn
import torch.nn.functional as F
from vajra.utils import LOGGER
from .convnextv1 import ConvNeXtV1
from .convnextv2 import ConvNeXtV2
from vajra.nn.modules import Linear
from vajra.nn.head import Classification, Detection

def build_convnext(in_channels,
                   num_classes,
                   size="small",
                   task="classify",
                   kpt_shape=None,
                   version="v2",
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
        model = ConvNeXtV2(in_channels, channels_list, num_blocks, num_classes=num_classes) if version == "v2" else ConvNeXtV1(in_channels, channels_list, num_blocks, num_classes=num_classes)
        head = Linear(channels_list[-1], num_classes)
        layers = []
        layers.append(model)
        layers.append(head)
        convnext = nn.Sequential(*layers)
        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        if verbose:            
            LOGGER.info(f"Task: classify; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info(f"Building ConvNeXt{str(version[0].upper()) + str(version[1])} ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            idx_counter = 0
            
            for i, (name, module) in enumerate(model.named_children()):
                if name in ["downsample_layers", "stages"]:
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
            
            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{idx_counter:>3}.{str(-1):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nModel Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"ConvNeXt{str(version[0].upper()) + str(version[1])} {size} Task: classify; Total Parameters: {np_model}\n\n")

        return convnext, layers, np_model
    return
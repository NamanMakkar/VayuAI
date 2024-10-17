# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import torch
import torch.nn as nn
from pathlib import Path
from vajra.checks import check_suffix, check_requirements
from vajra.utils.downloads import attempt_download_asset
from vajra.nn.modules import VajraStemBlock, VajraMerudandaBhag1, VajraMerudandaBhag3, VajraMerudandaBhag4, VajraMerudandaBhag5, VajraMerudandaBhag6, VajraGrivaBhag1, VajraGrivaBhag2, VajraStambh, VajraMerudandaBhag2, VajraAttentionBlock, Sanlayan, ChatushtayaSanlayan, ConvBNAct, MaxPool, ImagePoolingAttention, AttentionBottleneck, AttentionBlock
from vajra.nn.head import Detection, OBBDetection, Segementation, Classification, PoseDetection, WorldDetection, Panoptic, DEYODetection
from vajra.nn.vajrav2 import VajraV2Model, VajraV2CLSModel
from vajra.nn.vajrav3 import VajraV3Model, VajraV3CLSModel
from vajra.nn.backbones.efficientnets.effnetv2 import build_effnetv2
from vajra.nn.backbones.efficientnets.effnetv1 import build_effnetv1
from vajra.nn.backbones.convnexts.build import build_convnext
from vajra.nn.backbones.vayuvahana.me_nest import build_me_nest
from vajra.nn.backbones.vayuvahana.mixconvnext import build_mixconvnext
from vajra.nn.backbones.resnets.resnet import build_resnet
from vajra.nn.backbones.mobilenets.build import build_mobilenet
from vajra.utils import LOGGER, HYPERPARAMS_CFG_DICT, HYPERPARAMS_CFG_KEYS
from vajra.utils.torch_utils import model_info, initialize_weights, fuse_conv_and_bn, time_sync, intersect_dicts, scale_img
from vajra.loss import DetectionLoss, DEYODetectionLoss, OBBLoss, SegmentationLoss, PoseLoss, ClassificationLoss, PanopticLoss

try:
    import thop
except ImportError:
    thop = None

class VajraV1Model(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 num_repeats=[2, 2, 2, 2, 2, 2, 2, 2],
                 ) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], [1, 3, 5, -1], -1, [1, 5, 3, -1], -1, [8, 10, -1], -1, [10, 12, -1], -1, [12, 14, 16]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaBhag1(channels_list[1], channels_list[1], num_repeats[0], True, 3, False, 0.25) # stride 4
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraMerudandaBhag1(channels_list[1], channels_list[2], num_repeats[1], True, 3, False, 0.25) # stride 8 VajraMerudandaBhag5(channels_list[1], channels_list[2], True, 1)
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraMerudandaBhag1(channels_list[2], channels_list[3], num_repeats[2], True, 3) #VajraMerudandaBhag2(channels_list[2], channels_list[3], num_repeats[2], True, 3, 2, 0.75) # stride 16 VajraMerudandaBhag5(channels_list[2], channels_list[3], True, 2)
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraMerudandaBhag1(channels_list[3], channels_list[4], num_repeats[3], True, 3) #VajraMerudandaBhag2(channels_list[3], channels_list[4], num_repeats[3], True, 3, 2, 0.75) # stride 32 VajraMerudandaBhag5(channels_list[3], channels_list[4], True, 4)
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2, expansion_ratio=1.0)

        # Neck
        self.fusion4cbam = ChatushtayaSanlayan(in_c=channels_list[1:5], out_c=channels_list[6], expansion_ratio=0.5)
        self.vajra_neck1 = VajraGrivaBhag1(channels_list[6], num_repeats[4], 1)

        self.fusion4cbam2 = ChatushtayaSanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[6]], out_c=channels_list[8], expansion_ratio=0.5)
        self.vajra_neck2 = VajraGrivaBhag1(channels_list[8], num_repeats[5], 1)

        self.pyramid_pool_neck1 = Sanlayan(in_c=[channels_list[4], channels_list[6], channels_list[8]], out_c=channels_list[10], stride=2, expansion_ratio=0.5)
        self.vajra_neck3 = VajraGrivaBhag1(channels_list[10], num_repeats[6], 1)

        self.pyramid_pool_neck2 = Sanlayan(in_c=[channels_list[6], channels_list[8], channels_list[10]], out_c=channels_list[12], stride=2, expansion_ratio=0.5)
        self.vajra_neck4 = VajraGrivaBhag1(channels_list[12], num_repeats[7], 1)

    def forward(self, x):
        # Backbone
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        pool1 = self.pool1(vajra1)
        vajra2 = self.vajra_block2(pool1)

        pool2 = self.pool2(vajra2)
        vajra3 = self.vajra_block3(pool2)

        pool3 = self.pool3(vajra3)
        vajra4 = self.vajra_block4(pool3)
        pyramid_pool_backbone = self.pyramid_pool([vajra1, vajra2, vajra3, vajra4])

        # Neck
        fusion4 = self.fusion4cbam([vajra1, vajra2, vajra3, pyramid_pool_backbone])
        vajra_neck1 = self.vajra_neck1(fusion4)
        vajra_neck1 = vajra_neck1 + vajra3

        fusion4_2 = self.fusion4cbam2([vajra1, vajra3, vajra2, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_2)
        vajra_neck2 = vajra_neck2 + vajra2

        pyramid_pool_neck1 = self.pyramid_pool_neck1([pyramid_pool_backbone, vajra_neck1, vajra_neck2])
        vajra_neck3 = self.vajra_neck3(pyramid_pool_neck1)
        vajra_neck3 = vajra_neck3 + vajra3

        pyramid_pool_neck2 = self.pyramid_pool_neck2([vajra_neck1, vajra_neck2, vajra_neck3])
        vajra_neck4 = self.vajra_neck4(pyramid_pool_neck2)
        vajra_neck4 = vajra_neck4 + vajra4

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class VajraV1WorldModel(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256],
                 embed_channels=[256, 128, 256, 512],
                 num_heads = [8, 4, 8, 16],
                 num_repeats=[3, 6, 6, 3, 3, 3, 3, 3]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], [1, 3, 5, -1], -1, [1, 5, 3, -1], -1, [8, 10, -1], -1, [10, 12, -1], -1, [12, 14, 16]]
        # Backbone
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaBhag1(channels_list[1], channels_list[1], num_repeats[0], True, 3, False, 0.25) # stride 4
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraMerudandaBhag1(channels_list[1], channels_list[2], num_repeats[1], True, 3, False, 0.25) # stride 8
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraMerudandaBhag2(channels_list[2], channels_list[3], num_repeats[2], True, 3) # stride 16
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraMerudandaBhag2(channels_list[3], channels_list[4], num_repeats[3], True, 3) # stride 32
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2, expansion_ratio=1.0)
        # Neck
        self.fusion4cbam = ChatushtayaSanlayan(in_c=channels_list[1:5], out_c=channels_list[6], expansion_ratio=1.0)
        self.vajra_neck1 = VajraAttentionBlock(channels_list[5], channels_list[6], num_repeats[4], False, 1, embed_channels=embed_channels[0], num_heads=num_heads[0])

        self.fusion4cbam2 = ChatushtayaSanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[6]], out_c=channels_list[8])
        self.vajra_neck2 = VajraAttentionBlock(channels_list[7], channels_list[8], num_repeats[5], False, 1, embed_channels=embed_channels[1], num_heads=num_heads[1])

        self.pyramid_pool_neck1 = Sanlayan(in_c=[channels_list[4], channels_list[6], channels_list[8]], out_c=channels_list[9], stride=2, use_cbam=False)
        self.vajra_neck3 = VajraAttentionBlock(channels_list[9], channels_list[10], num_repeats[6], False, 1, embed_channels=embed_channels[2], num_heads=num_heads[2])

        self.pyramid_pool_neck2 = Sanlayan(in_c=[channels_list[6], channels_list[8], channels_list[10]], out_c=channels_list[11], stride=2, use_cbam=False)
        self.vajra_neck4 = VajraAttentionBlock(channels_list[11], channels_list[12], num_repeats[7], False, 1, embed_channels=embed_channels[3], num_heads=num_heads[3])

    def forward(self, x):
        # Backbone
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        pool1 = self.pool1(vajra1)
        vajra2 = self.vajra_block2(pool1)

        pool2 = self.pool2(vajra2)
        vajra3 = self.vajra_block3(pool2)

        pool3 = self.pool3(vajra3)
        vajra4 = self.vajra_block4(pool3)
        pyramid_pool_backbone = self.pyramid_pool([vajra1, vajra2, vajra3, vajra4])

        # Neck
        fusion4 = self.fusion4cbam([vajra1, vajra2, vajra3, pyramid_pool_backbone])
        vajra_neck1 = self.vajra_neck1(fusion4)

        fusion4_2 = self.fusion4cbam2([vajra1, vajra3, vajra2, vajra_neck1])
        vajra_neck2 = self.vajra_neck2(fusion4_2)

        pyramid_pool_neck1 = self.pyramid_pool_neck1([pyramid_pool_backbone, vajra_neck1, vajra_neck2])
        vajra_neck3 = self.vajra_neck3(pyramid_pool_neck1)

        pyramid_pool_neck2 = self.pyramid_pool_neck2([vajra_neck1, vajra_neck2, vajra_neck3])
        vajra_neck4 = self.vajra_neck4(pyramid_pool_neck2)

        outputs = [vajra_neck2, vajra_neck3, vajra_neck4]
        return outputs

class VajraV1CLSModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels_list=[64, 128, 256, 512, 1024],
                 num_repeats=[3, 6, 6, 3]) -> None:
        super().__init__()
        self.from_list = [-1, -1, -1, -1, -1, -1, -1, -1, [1, 3, 5, -1], -1]
        self.stem = VajraStambh(in_channels, channels_list[0], channels_list[1])
        self.vajra_block1 = VajraMerudandaBhag1(channels_list[1], channels_list[1], num_repeats[0], True, 3, False, 0.25) # stride 4
        self.pool1 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block2 = VajraMerudandaBhag1(channels_list[1], channels_list[2], num_repeats[1], True, 3, False, 0.25) # stride 8
        self.pool2 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block3 = VajraMerudandaBhag2(channels_list[2], channels_list[3], num_repeats[2], True, 3) # stride 16
        self.pool3 = MaxPool(kernel_size=2, stride=2)
        self.vajra_block4 = VajraMerudandaBhag2(channels_list[3], channels_list[4], num_repeats[3], True, 3) # stride 32
        self.pyramid_pool = Sanlayan(in_c=[channels_list[1], channels_list[2], channels_list[3], channels_list[4]], out_c=channels_list[4], stride=2, expansion_ratio=1.0)

    def forward(self, x):
        stem = self.stem(x)
        vajra1 = self.vajra_block1(stem)

        pool1 = self.pool1(vajra1)
        vajra2 = self.vajra_block2(pool1)

        pool2 = self.pool2(vajra2)
        vajra3 = self.vajra_block3(pool2)

        pool3 = self.pool3(vajra3)
        vajra4 = self.vajra_block4(pool3)
        pyramid_pool_backbone = self.pyramid_pool([vajra1, vajra2, vajra3, vajra4])

        return pyramid_pool_backbone

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

def build_vajra(in_channels,
                task,
                num_classes,
                size="nano",
                version="v1",
                verbose=False,
                world=False,
                kpt_shape=None,
                model_name="vajra-v1-nano-det",
            ):
    
    stride = torch.tensor([8., 16., 32.])
    config_dict = {"nano": [0.33, 0.25, 1024], 
                   "small": [0.33, 0.5, 1024], 
                   "medium": [0.67, 0.75, 512], 
                   "large": [1.0, 1.0, 512], 
                   "xlarge": [1.0, 1.5, 512],
                }
    depth_mul = config_dict[size][0]
    width_mul = config_dict[size][1]
    num_protos = 256 * width_mul
    max_channels = config_dict[size][2]
    max_channels = make_divisible(max_channels, 8)

    #if version != "v3":
        #num_repeats = [3, 6, 6, 3, 3, 3, 3, 3] if task != "classify" else [3, 6, 6, 3]
        #channels_list = [64, 128, 256, 512, 1024, 256, 256, 256, 256, 256, 256, 256, 256] if task != "classify" else [64, 128, 256, 512, 1024]
    #else:
        #num_repeats = [3, 3, 3, 3, 3, 3, 3, 3] if task != "classify" else [3, 6, 6, 3]
        #channels_list = [64, 128, 256, 512, 1024, 256, 512, 256, 256, 256, 512, 512, 1024] if task != "classify" else [64, 128, 256, 512, 1024]
    
    num_repeats = [2, 2, 2, 2, 2, 2, 2, 2] if task != "classify" else [3, 3, 3, 3]
    channels_list = [64, 128, 256, 512, 1024, 256, 512, 256, 256, 256, 512, 512, 1024] if task != "classify" else [64, 128, 256, 512, 1024]
    channels_list = [make_divisible(min(ch, max_channels) * width_mul, 8) for ch in channels_list]
    num_repeats = [(max(round(n * depth_mul), 1) if n > 1 else n) for n in num_repeats]

    embed_channels = [256, 128, 256, 512]
    embed_channels = [make_divisible(min(ch, max_channels // 2) * width_mul, 8) for ch in embed_channels]
    num_heads = [8, 4, 8, 16]
    num_heads = [int(max(round(min(nh, max_channels // 2 // 32)) * width_mul, 1)) if nh > 1 else 1 for nh in num_heads]

    if task != "classify":
        if task != "world":
            if version == "v1":
                model = VajraV1Model(in_channels, channels_list, num_repeats)
            elif version == "v2": 
                model = VajraV2Model(in_channels, channels_list, num_repeats)
            elif version == "v3":
                model = VajraV3Model(in_channels, channels_list, num_repeats)

            head_channels = [channels_list[8], channels_list[10], channels_list[12]]

            if task == "detect":
                if model_name.split("-")[1] == "deyo":
                    head = DEYODetection(num_classes=num_classes, in_channels=head_channels)
                else:
                    head = Detection(num_classes, head_channels)
            elif task == "segment":
                head = Segementation(num_classes, in_channels=head_channels, num_protos=num_protos)
            elif task == "pose":
                head = PoseDetection(num_classes, in_channels=head_channels, keypoint_shape=kpt_shape if any(kpt_shape) else (17, 3))
            elif task == "obb":
                head = OBBDetection(num_classes, in_channels=head_channels)
            elif task == "panoptic":
                head = Panoptic(num_classes, in_channels=head_channels, num_protos=num_protos)

        else:
            model = VajraV1WorldModel(in_channels, channels_list, embed_channels, num_heads, num_repeats)
            head_channels = channels_list[-3:]
            head = WorldDetection(num_classes, embed_dim=head_channels[-1], with_bn=True, in_channels=head_channels)
        
        layers = []
        np_model = sum(x.numel() for x in model.parameters())
        np_head = sum(x.numel() for x in head.parameters())

        if verbose:
            LOGGER.info(f"Task: {task}; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building Vajra ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            for i, (name, module) in enumerate(model.named_children()):
                np = sum(x.numel() for x in module.parameters())
                md_info, arg_info = module.get_module_info()
                LOGGER.info(f"{i:>3}.{str(model.from_list[i]):>20}{np:>10}  {md_info:<45}{arg_info:<30}")
            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{i+1:>3}.{str(model.from_list[-1]):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nBackbone and Neck Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"VajraV1-{size}; Task: {task}; Total Parameters: {np_model + np_head}\n\n")

        np_model += np_head
        head.stride = stride
        head.bias_init()
        layers.append(model)
        layers.append(head) # List of backbone+neck and head in order to access the head easily
        
        vajra = nn.Sequential(*layers) # nn.Sequential model used for train, val, pred

        return vajra, stride, layers, np_model

    else:
        model = VajraV1CLSModel(in_channels=3, channels_list=channels_list, num_repeats=num_repeats) if version == "v1" else VajraV2CLSModel(in_channels=3, channels_list=channels_list, num_repeats=num_repeats)
        np_model = sum(x.numel() for x in model.parameters())
        head = Classification(in_c=channels_list[-1], out_c=num_classes)
        np_head = sum(x.numel() for x in head.parameters())
        layers=[]
        layers.append(model)
        layers.append(head)
        vajra = nn.Sequential(*layers)
        np_model += np_head
        if verbose:
            LOGGER.info(f"Task: {task}; Number of Classes: {num_classes}\n\n\n")
            LOGGER.info("Building Vajra ...\n\n")
            LOGGER.info(f"\n{'index':>3}{'from':>20}{'params':>10}  {'module':<45}{'arguments':<30}\n")
            for i, (name, module) in enumerate(model.named_children()):
                np = sum(x.numel() for x in module.parameters())
                md_info, arg_info = module.get_module_info()
                LOGGER.info(f"{i:>3}.{str(model.from_list[i]):>20}{np:>10}  {md_info:<45}{arg_info:<30}")
            head_md_info, head_arg_info = head.get_module_info()
            LOGGER.info(f"{i+1:>3}.{str(model.from_list[-1]):>20}{np_head:>10}  {head_md_info:<45}{head_arg_info:<30}")
            LOGGER.info(f"\nBackbone Parameters: {np_model}\n\n")
            LOGGER.info(f"Head Parameters: {np_head}\n\n")
            LOGGER.info(f"Vajra{str(version[0].upper()) + str(version[1])} {size}; Task: {task}; Total Parameters: {np_model}\n\n")

    return vajra, stride, layers, np_model

class Model(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False):
        if augment:
            return self._predict_augment(x)

        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        dt = []

        for layer in self.model:
            if profile:
                self._profile_one_layer(layer, x, dt)
            x = layer(x)
        return x

    def _predict_augment(self, x):
        LOGGER.warning(
            f"WARNING! Augmented Inference is task specific"
            f"Switching to single-scale inference"
        )
        return self._predict_once(x)

    def _profile_one_layer(self, layer, x, dt):
        num_params_layer = sum(x.numel() for x in layer.parameters())
        copy_input = layer == self.combined_model[-1] and isinstance(x, list)
        flops = thop.profile(layer, inputs=[x.copy() if copy_input else x], verbose=False)[0]
        t = time_sync()
        for _ in range(10):
            layer(x.copy() if copy_input else x)
        dt.append((time_sync() - t) * 100)
        if layer == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {num_params_layer:10.0f}  {str(layer)}')
        if copy_input:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, ConvBNAct) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)
        return self
    
    def is_fused(self, threshold=10):
        batchnorms = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, batchnorms) for v in self.modules()) < threshold

    def info(self, detailed = False, verbose=True, img_size=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=img_size)

    def _apply(self, fn):
        self = super()._apply(fn)
        head = self.head
        if isinstance(head, (Detection)):
            head.stride = fn(head.stride)
            head.anchors = fn(head.anchors)
            head.strides = fn(head.strides)
        return self

    def load(self, weights, verbose=True):
        model = weights['model'] if isinstance(weights, dict) else weights
        fp32_checkpoint_state_dict = model.float().state_dict()
        fp32_checkpoint_state_dict = intersect_dicts(fp32_checkpoint_state_dict, self.model.state_dict())
        self.model.load_state_dict(fp32_checkpoint_state_dict, strict=False)
        if verbose:
            LOGGER.info(f'Transferred {len(fp32_checkpoint_state_dict)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError('Loss computation is task specific')

class DetectionModel(Model):
    def __init__(self, model_name='vajra-v1-nano-det', channels=3, num_classes=None, verbose=True, world=False, kpt_shape=None) -> None:
        super().__init__()
        size = model_name.split("-")[-2]
        task = get_task(model_name)
        self.num_classes = 80
        self.model_name = model_name
        if "vajra" in model_name:
            version = model_name.split("-")[-3]
        if num_classes and num_classes != self.num_classes:
            LOGGER.info(f"Overriding num_classes={self.num_classes} with num_classes={num_classes}")
            self.num_classes = num_classes
        
        if "vajra_effnetv1" in model_name:
            self.model, self.stride, self.layers, self.np_model = build_effnetv1(in_channels=channels, num_classes=self.num_classes, size=size, task=task, verbose=verbose, kpt_shape=kpt_shape)
        elif "vajra_effnetv2" in model_name:
            self.model, self.stride, self.layers, self.np_model = build_effnetv2(in_channels=channels, num_classes=self.num_classes, size=size, task=task, verbose=verbose, kpt_shape=kpt_shape)
        else:
            self.model, self.stride, self.layers, self.np_model = build_vajra(in_channels=channels, task=task, num_classes=self.num_classes, size=size, version=version, verbose=verbose, world=world, kpt_shape=kpt_shape, model_name=model_name)
        #LOGGER.info(f"Head Layer {self.model[-1].get_module_info()}\n")
        self.head = self.model[-1]
        self.task = task
        self.names = {i: f"{i}" for i in range(self.num_classes)}
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        img_size = x.shape[-2:]
        scales = [1, 0.83, 0.67]
        flips = [None, 3, None]
        oup = []
        for scale, flip in zip(scales, flips):
            xi = scale_img(x.flip(flip) if flip else x, scale, gs=int(self.stride.max()))
            o = super().predict(xi)[0]
            o = self._descale_pred(o, flip, scale, img_size)
            oup.append(o)
        oup = self._clip_augmented(oup)
        return torch.cat(oup, -1), None

    @staticmethod
    def _descale_pred(pred, flip, scale, img_size, dim=1):
        pred[:, :4] /= scale
        x, y, wh, cls = pred.split((1, 1, 2, pred.shape[dim] - 4), dim)
        if flip == 2:
            y = img_size[0] - y
        if flip == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, oup):
        num_layers = self.head.num_det_layers
        grid_pts = sum(4 ** x for x in range(num_layers))
        num_exclude = 1
        indices = (oup[0].shape[-1] // grid_pts) * sum(4 ** x for x in range(num_exclude))
        oup[0] = oup[0][..., :-indices]
        indices = (oup[-1].shape[-1] // grid_pts) * sum(4 ** (num_layers - 1 - x) for x in range(num_exclude))
        oup[-1] = oup[-1][..., indices:]

    def init_criterion(self):
        return DetectionLoss(self)

class SegmentationModel(DetectionModel):
    def __init__(self, model_name='vajra-v1-nano-seg', channels=3, num_classes=None, verbose=True):
        super().__init__(model_name, channels, num_classes, verbose)

    def init_criterion(self):
        return SegmentationLoss(self)

class OBBModel(DetectionModel):
    def __init__(self, model_name='vajra-v1-nano-obb', channels=3, num_classes=None, verbose=True, task="obb"):
        super().__init__(model_name, channels, num_classes, verbose)

    def init_criterion(self):
        return OBBLoss(self)

class PoseModel(DetectionModel):
    def __init__(self, model_name='vajra-v1-nano-pose', channels=3, num_classes=None, data_kpt_shape=(None, None), verbose=True):
        self.kpt_shape = data_kpt_shape
        super().__init__(model_name, channels, num_classes, verbose, kpt_shape=data_kpt_shape)

    def init_criterion(self):
        return PoseLoss(self)

class PanopticModel(DetectionModel):
    def __init__(self, model_name='vajra-v1-nano-panoptic', channels=3, num_classes=None, verbose=True, world=False, kpt_shape=None) -> None:
        super().__init__(model_name, channels, num_classes, verbose, world, kpt_shape)

    def init_criterion(self):
        return PanopticLoss(self)

class VajraDEYODetectionModel(DetectionModel):
    def init_criterion(self):
        return DEYODetectionLoss(self)

    def predict(self, x, profile=False, visualize=False, augment=False):
        imgsz = x.shape[2:]
        dt = []
        y = []
        for layer in self.model[:-1]:
            if profile:
                self._profile_one_layer(layer, x, dt)
            x = layer(x)
            y.append(x)

        head = self.model[-1]
        x = head(y[-1], imgsz)
        return x

class ClassificationModel(Model):
    def __init__(self, model_name='vajra-v1-nano-cls', channels=3, num_classes=None, verbose=True) -> None:
        super().__init__()
        if "convnext" or "effnetv2" or "effnetv1" in model_name:
            size = model_name.split("-")[-1]
        else:
            size = model_name.split("-")[-2]
        #task = get_task(model_name)
        self.num_classes = 1000
        if num_classes and num_classes != self.num_classes:
            LOGGER.info(f"Overriding num_classes={self.num_classes} with num_classes={num_classes}")
            self.num_classes = num_classes
        self.model_name = model_name

        if "convnext" in model_name:
            version = model_name[8:10]
            self.model, self.layers, self.np_model = build_convnext(channels, num_classes=self.num_classes, size=size, task="classify", version=version, verbose=verbose)
        elif "effnetv2" in model_name:
            self.model, self.layers, self.np_model = build_effnetv2(channels, num_classes=self.num_classes, size=size, task="classify", verbose=verbose)
        elif "effnetv1" in model_name:
            self.model, self.layers, self.np_model = build_effnetv1(channels, num_classes=self.num_classes, size=size, task="classify", verbose=verbose)
        else:
            self.model, _, self.layers, self.np_model = build_vajra(channels, task="classify", num_classes=self.num_classes, size=size, verbose=verbose)
        
        self.head = self.layers[-1]
        self.stride = torch.Tensor([1])
        self.names = {i: f'{i}' for i in range(self.num_classes)}
        self.task = "classify"

    @staticmethod
    def reshape_outputs(model, num_classes):
        name, head = list((model.model if hasattr(model, 'model') else model).named_children())[-1]

        if isinstance(head, Classification):
            if head.linear.out_features != num_classes:
                head.linear = nn.Linear(head.linear.in_features, num_classes)
        elif isinstance(head, nn.Linear):
            if head.out_features != num_classes:
                setattr(model, name, nn.Linear(head.in_features, num_classes))
        elif isinstance(head, nn.Sequential):
            types = [(type(x) for x in head)]
            if nn.Linear in types:
                idx = types.index(nn.Linear)
                if head[idx].out_features != num_classes:
                    head[idx] = nn.Linear(head[idx].in_features, num_classes)
            elif nn.Conv2d in types:
                idx = types.index(nn.Conv2d)
                if head[idx].out_channels != num_classes:
                    head[idx] = nn.Conv2d(head[idx].in_channels, num_classes, head[idx].kernel_size, head[idx].stride, bias=head[idx].bias is not None)

    def init_criterion(self):
        return ClassificationLoss()

class Ensemble(nn.ModuleList):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        oups = [module(x, augment, profile, visualize)[0] for module in self]
        oups = torch.cat(oups, 2)
        return oups, None

class VajraWorld(DetectionModel):
    def __init__(self, model_name='vajra-v1-nano-world-det', channels=3, num_classes=None, verbose=True) -> None:
        self.txt_feats = torch.randn(1, num_classes or 80, 512)
        self.clip_model = None
        super().__init__(model_name, channels, num_classes, verbose, world=True)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        
        if (not getattr(self, "clip_model", None) and cache_clip_model):
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.head.num_classes = len(text)

    def predict(self, x, profile=False, visualize=False, augment=False):
        txt_feats = (self.txt_feats if txt_feats is not None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        orig_txt_feats = txt_feats.clone()
        dt = []

        for layer in self.model:
            if profile:
                self._profile_one_layer(layer, x, dt)
            if isinstance(layer, VajraAttentionBlock):
                x = layer(x, txt_feats)
            if isinstance(layer, WorldDetection):
                x = layer(x, orig_txt_feats)
            if isinstance(layer, ImagePoolingAttention):
                txt_feats = layer(x, txt_feats)
            else:
                x = layer(x)
        return x

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        
        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)

def torch_safe_load(weight):
    from vajra.utils.downloads import attempt_download_vajra, attempt_download_asset
    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)

    try:
        checkpoint = torch.load(file, map_location='cpu')
    except ModuleNotFoundError as e:
        LOGGER.warning(
            f"WARNING! {weight} appears to require '{e.name}'"
            f"\nRecommended fixes are to train a new model from scratch using the latest 'vajra' package or to"
            f"run a command with an official Vajra model i.e 'vajra predict model=vajra-v1-nano.pt'" 
        )
    
    if not isinstance(checkpoint, dict):
        LOGGER.warning(
            f"WARNING! The file '{weight}' appears to be improperly saved or formatted."
            f"For optimal results, use model.save('filename.pt') to correctly save Vajra models."
        )
        checkpoint = {"model": checkpoint.model}

    return checkpoint, file

def load_weight(weight, device=None, inplace=True, fuse=False):
    checkpt, weight = torch_safe_load(weight)
    args = {**HYPERPARAMS_CFG_DICT, **(checkpt.get("train_args", {}))}
    model = (checkpt.get("ema") or checkpt["model"]).to(device).float()
    model.args = {k:v for k, v in args.items() if k in HYPERPARAMS_CFG_KEYS}
    model.pt_path = weight

    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])
    
    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()

    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = inplace
        elif isinstance(module, nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    
    return model, checkpt

def load_ensemble_weights(weights, device = None, inplace = True, fuse = False):
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        checkpoint, w = torch_safe_load(w)
        args = {**HYPERPARAMS_CFG_DICT, **checkpoint["train_args"]} if "train_args" in checkpoint else None
        model = (checkpoint.get("ema") or checkpoint['model']).to(device).float()

        model.args = args
        model.pt_path = w
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())

    for module in ensemble.modules():
        if hasattr(module, "inplace"):
            module.inplace = inplace
        elif isinstance(module, nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None

    if len(ensemble) == 1:
        return ensemble[-1]

    LOGGER.info(f'Ensemble created with {weights}\n')

    for k in "names", "num_classes", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([module.stride.max() for module in ensemble])))].stride
    assert all(ensemble[0].num_classes == module.num_classes for module in ensemble), f"Models differ in class counts {[module.num_classes for module in ensemble]}"
    return ensemble

def get_task(model):
    if isinstance(model, nn.Module):
        for module in model.modules():
            if isinstance(module, Detection):
                return "detect"
            elif isinstance(module, Segementation):
                return "segment"
            elif isinstance(module, Classification):
                return "classify"
            elif isinstance(module, PoseDetection):
                return "pose"
            elif isinstance(module, OBBDetection):
                return "obb"
            elif isinstance(module, WorldDetection):
                return "world"
    if isinstance(model, (str, Path)):
        model_path = Path(model)
        if "-det" in model_path.stem:
            return "detect"
        elif "-cls" in model_path.stem:
            return "classify"
        elif "-seg" in model_path.stem:
            return "segment"
        elif "-pose" in model_path.stem:
            return "pose"
        elif "-obb" in model_path.stem:
            return "obb"
        elif "-world" in model_path.stem:
            return "world"

    LOGGER.warning('WARNING! Unable to estimate model task, assuming "task = detect"')
    return "detect"      
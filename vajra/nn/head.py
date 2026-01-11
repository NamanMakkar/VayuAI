# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import OrderedDict, List
from torch.nn.init import xavier_uniform_, constant_ 
from vajra.utils.dfine_ops import get_contrastive_denoising_training_group
from vajra.tal.anchor_generator import dist2bbox, dist2rbox, generate_anchors
from vajra.nn.modules import DepthwiseConvBNAct, ConvMakkarNorm, Squeeze_Excite_Layer, ConvBNAct, DistributedFocalLoss, ProtoMaskModule, UConv, BNContrastiveHead, ContrastiveHead, AnchorMLP, ImagePoolingAttention, RepVGGDW, Residual, SwiGLUFFN2, SAVPE
from vajra.nn.transformer import ScaleAdaptiveDecoderLayer, ScaleAdaptiveTransformerDecoder, MLP, HybridEncoder, TransformerDecoder, TransformerDecoderLayer, Integral
from vajra.utils import LOGGER, NOT_MACOS14
from vajra.utils.torch_utils import smart_inference_mode, fuse_conv_and_bn
from vajra.nn.utils import bias_init_with_prob

class Classification(nn.Module):
    export = False
    def __init__(self, in_c, out_c, hidden_c=1280, kernel_size=1, stride=1, padding=None, groups=1) -> None:
        super().__init__()
        self.hidden_c = hidden_c
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.conv = ConvBNAct(self.in_c, 
                              self.hidden_c, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              groups=groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.hidden_c, out_c)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear((self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)

    def get_module_info(self):
        return f"Classification", f"[{self.in_c}, {self.out_c}, {self.hidden_c}, {self.kernel_size}, {self.stride}, {self.padding}, {self.groups}]"

class Detection(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, num_classes=80, in_channels=[]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_det_layers = len(in_channels)
        self.reg_max = 16
        self.num_outputs_per_anchor = self.num_classes + self.reg_max * 4
        self.stride = torch.zeros(self.num_det_layers)
        c2 = max((8, in_channels[0] // 4, self.reg_max * 4))
        c3 = max(in_channels[0], min(self.num_classes, 100))
        self.branch_det = nn.ModuleList(
            nn.Sequential(ConvBNAct(ch, c2, kernel_size=3),
                          ConvBNAct(c2, c2, kernel_size=3),
                          nn.Conv2d(c2, 4*self.reg_max, 1))
                          for ch in in_channels
        )
        self.branch_cls = (
            nn.ModuleList(nn.Sequential(ConvBNAct(ch, c3, 1, 3), ConvBNAct(c3, c3, 1, 3), nn.Conv2d(c3, self.num_classes, 1)) for ch in in_channels) 
            if self.legacy
            else
            nn.ModuleList(nn.Sequential(nn.Sequential(DepthwiseConvBNAct(ch, ch, 1, 3), ConvBNAct(ch, c3, 1, 1)),
                                        nn.Sequential(DepthwiseConvBNAct(c3, c3, 1, 3), ConvBNAct(c3, c3, 1, 1)),
                                        nn.Conv2d(c3, self.num_classes, 1)
                                        )
                            for ch in in_channels
            )
        )
        self.distributed_focal_loss = DistributedFocalLoss(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def bias_init(self):
        detection_module = self
        for branch_a, branch_b, stride in zip(detection_module.branch_det, detection_module.branch_cls, detection_module.stride):
            branch_a[-1].bias.data[:] = 1.0
            branch_b[-1].bias.data[:detection_module.num_classes] = math.log( 5 / detection_module.num_classes / (640 / stride) ** 2)

    def forward(self, x):
        shape = x[0].shape

        for i in range(self.num_det_layers):
            x[i] = torch.cat((self.branch_det[i](x[i]), self.branch_cls[i](x[i])), 1)

        if self.training:
            return x

        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in generate_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.num_outputs_per_anchor, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]

        else:
            box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        
        if self.export and self.format in ("tflite", "edgetpu"):
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dist_box = self.decode_bboxes(self.distributed_focal_loss(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])

        else:
            dist_box = self.decode_bboxes(self.distributed_focal_loss(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dist_box, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchor_points=anchors, xywh=True, dim=1)

    def get_module_info(self):
        return f"Detection", f"[{self.num_classes}, {self.in_channels}]"

class Panoptic(Detection):
    def __init__(self, num_classes=80, num_masks=32, sem_nc=93, num_protos=256, in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.num_masks = num_masks
        self.num_protos = num_protos
        self.sem_nc = sem_nc
        self.proto = ProtoMaskModule(in_channels[0], self.num_protos, self.num_masks)
        self.uconv = UConv(in_channels[0], in_channels[0]//4, self.sem_nc + self.num_classes)
        self.detection = Detection.forward
        c4 = max(in_channels[0] // 4, self.num_masks)
        self.branch_seg = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(x, c4, kernel_size=3),
                nn.Conv2d(c4, self.num_masks, 1)
            )
            for x in in_channels
        )

    def forward(self, x):
        proto_masks = self.proto(x[0])
        s = self.uconv(x[0])
        batch_size = proto_masks.shape[0]

        mask_coefficients = torch.cat([self.branch_seg[i](x[i]).view(batch_size, self.num_masks, -1) for i in range(self.num_det_layers)], 2)
        x = self.detection(self, x)
        if self.training:
            return x, mask_coefficients, proto_masks, s

        return (torch.cat([x, mask_coefficients], 1), proto_masks, s) if self.export else (torch.cat([x[0], mask_coefficients], 1), (x[1], mask_coefficients, proto_masks, s))

    def get_module_info(self):
        return f"Panoptic", f"[{self.num_classes}, {self.num_masks}, {self.sem_nc}, {self.num_protos}, {self.in_channels}]"

class Segmentation(Detection):
    def __init__(self, num_classes=80, num_masks=32, num_protos=256, in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.num_masks = num_masks
        self.num_protos = num_protos
        self.proto = ProtoMaskModule(in_channels[0], self.num_protos, self.num_masks)
        c4 = max(in_channels[0] // 4, self.num_masks)
        self.branch_seg = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(ch, c4, kernel_size=3, stride=1),
                ConvBNAct(c4, c4, kernel_size=3, stride=1),
                nn.Conv2d(c4, self.num_masks, 1)
            )
            for ch in in_channels
        )

    def forward(self, x):
        proto_masks = self.proto(x[0])
        batch_size = proto_masks.shape[0]

        mask_coefficients = torch.cat([self.branch_seg[i](x[i]).view(batch_size, self.num_masks, -1)
                                       for i in range(self.num_det_layers)], 2)
        x = Detection.forward(self, x)
        if self.training:
            return x, mask_coefficients, proto_masks
        return (torch.cat([x, mask_coefficients], 1), proto_masks) if self.export else (torch.cat([x[0], mask_coefficients], 1), (x[1], mask_coefficients, proto_masks))

    def get_module_info(self):
        return f"Segmentation", f"[{self.num_classes}, {self.num_masks}, {self.num_protos}, {self.in_channels}]"

class PoseDetection(Detection):
    def __init__(self, num_classes=80, keypoint_shape=(17, 3), in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.keypoint_shape = keypoint_shape
        self.num_keypoints = keypoint_shape[0] * keypoint_shape[1]
        self.detection =  Detection.forward
        c4 = max(in_channels[0] // 4, self.num_keypoints)
        self.branch_pose_detect = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(ch, c4, kernel_size=3, stride=1),
                ConvBNAct(c4, c4, kernel_size=3, stride=1),
                nn.Conv2d(c4, self.num_keypoints, 1)
            )
            for ch in in_channels
        )

    def decode_keypoints(self, batch_size, keypoints):
        ndim = self.keypoint_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:
                y = keypoints.view(batch_size, *self.keypoint_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                y = keypoints.view(batch_size, *self.keypoint_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(batch_size, self.num_keypoints, -1)
        else:
            y = keypoints.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid_()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y
    
    def forward(self, x):
        batch_size = x[0].shape[0] # batch_size
        keypoint = torch.cat([self.branch_pose_detect[i](x[i]).view(batch_size, self.num_keypoints, -1) 
                              for i in range(self.num_det_layers)], -1) # (batch_size, 17*3, h*w)
        x = Detection.forward(self, x)
        if self.training:
            return x, keypoint
        pred_keypoints = self.decode_keypoints(batch_size, keypoint)
        return torch.cat([x, pred_keypoints], 1) if self.export else (torch.cat([x[0], pred_keypoints], 1), (x[1], keypoint))

    def get_module_info(self):
        return f"PoseDetection", f"[{self.num_classes}, {self.keypoint_shape}, {self.in_channels}]"

class SmallObjSegmentation(Detection):
    def __init__(self, num_classes=80, keypoint_shape=(17, 3), num_masks=32, num_protos=256, in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.keypoint_shape = keypoint_shape
        self.num_keypoints = keypoint_shape[0] * keypoint_shape[1]
        c4 = max(in_channels[0] // 4, self.num_keypoints)
        self.branch_pose_detect = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(ch, c4, kernel_size=3, stride=1),
                ConvBNAct(c4, c4, kernel_size=3, stride=1),
                nn.Conv2d(c4, self.num_keypoints, 1)
            )
            for ch in in_channels
        )
        
        self.num_masks = num_masks
        self.num_protos = num_protos
        c5 = max(in_channels[0] // 4, self.num_masks)
        self.proto = ProtoMaskModule(in_channels[0], self.num_protos, self.num_masks)
        self.branch_seg = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(ch, c5, kernel_size=3, stride=1),
                ConvBNAct(c5, c5, kernel_size=3, stride=1),
                nn.Conv2d(c5, self.num_masks, 1)
            )
            for ch in in_channels
        )

    def decode_keypoints(self, batch_size, keypoints):
        ndim = self.keypoint_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:
                y = keypoints.view(batch_size, *self.keypoint_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                y = keypoints.view(batch_size, *self.keypoint_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(batch_size, self.num_keypoints, -1)
        else:
            y = keypoints.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid_()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y
    
    def forward(self, x):
        proto_masks = self.proto(x[0])
        batch_size = proto_masks.shape[0]
        keypoint = torch.cat([self.branch_pose_detect[i](x[i]).view(batch_size, self.num_keypoints, -1) 
                              for i in range(self.num_det_layers)], -1)
        mask_coefficients = torch.cat([self.branch_seg[i](x[i]).view(batch_size, self.num_masks, -1)
                                       for i in range(self.num_det_layers)], 2)
        x = Detection.forward(self, x)
        if self.training:
            return x, keypoint, mask_coefficients, proto_masks
        pred_keypoints = self.decode_keypoints(batch_size, keypoint)
        return (torch.cat([x, pred_keypoints, mask_coefficients], 1), proto_masks) if self.export else (torch.cat([x[0], pred_keypoints, mask_coefficients], 1), (x[1], keypoint, proto_masks))
    
    def get_module_info(self):
        return f"SmallObjSegmentation", f"[{self.num_classes}, {self.keypoint_shape}, {self.num_masks}, {self.num_protos}, {self.in_channels}]"

class WorldDetection(Detection):
    def __init__(self, num_classes=80, embed_dim=512, with_bn=False, in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.embed_dim = embed_dim
        self.with_bn = with_bn
        self.in_channels = in_channels
        c3 = max(in_channels[0], min(self.num_classes, 100))
        self.branch3 = nn.ModuleList(nn.Sequential(ConvBNAct(ch, c3, stride=1, kernel_size=3), ConvBNAct(c3, c3, kernel_size=3, stride=1), nn.Conv2d(c3, embed_dim, 1)) for ch in in_channels)
        self.branch4 = nn.ModuleList(BNContrastiveHead(embed_dim) if with_bn else ContrastiveHead() for _ in in_channels)

    def forward(self, x, text):
        for i in range(self.num_det_layers):
            x[i] = torch.cat((self.branch_det[i](x[i]), self.branch4[i](self.branch3[i](x[i]), text)), 1)
        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.num_classes + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in generate_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]

        else:
            box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        
        if self.export and self.format in ("tflite", "edgetpu"):
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dist_box = self.decode_bboxes(self.distributed_focal_loss(box), self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dist_box = self.decode_bboxes(self.distributed_focal_loss(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dist_box, cls.sigmoid()), 1)

        return y if self.export else (y, x)
    
    def bias_init(self):
        detection_module = self
        for branch_a, branch_b, stride in zip(detection_module.branch_det, detection_module.branch_cls, detection_module.stride):
            branch_a[-1].bias.data[:] = 1.0

    def get_module_info(self):
        return f"WorldDetection", f"[{self.num_classes}, {self.embed_dim}, {self.with_bn}, {self.in_channels}]"


class OBBDetection(Detection):
    def __init__(self, num_classes=80, num_extra_params=1, in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.num_extra = num_extra_params
        c4 = max(in_channels[0] // 4, self.num_extra)
        self.oriented_branch = nn.ModuleList(
            nn.Sequential(
            ConvBNAct(ch, c4, kernel_size=3, stride=1),
            ConvBNAct(c4, c4, kernel_size=3, stride=1), 
            nn.Conv2d(c4, self.num_extra, 1)
            ) 
            for ch in in_channels
        )

    def forward(self, x):
        batch_size = x[0].shape[0]
        angle = torch.cat([self.oriented_branch[i](x[i]).view(batch_size, self.num_extra, -1) for i in range(self.num_det_layers)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi

        if not self.training:
            self.angle = angle
        x = Detection.forward(self, x)
        if self.training:
            return x, angle
        
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchor_points=anchors, dim=1)

    def get_module_info(self):
        return f"OBBDetection", f"[{self.num_classes}, {self.num_extra}, {self.in_channels}]"

class SmallOBBDetection(Detection):
    def __init__(self, num_classes=80, num_extra_params=1, keypoint_shape=(17, 3), in_channels=[]) -> None:
        super().__init__(num_classes, in_channels)
        self.num_extra = num_extra_params
        c4 = max(in_channels[0] // 4, self.num_extra)
        self.oriented_branch = nn.ModuleList(
            nn.Sequential(
            ConvBNAct(ch, c4, kernel_size=3, stride=1),
            ConvBNAct(c4, c4, kernel_size=3, stride=1), 
            nn.Conv2d(c4, self.num_extra, 1)
            ) 
            for ch in in_channels
        )
        self.keypoint_shape = keypoint_shape
        self.num_keypoints = keypoint_shape[0] * keypoint_shape[1]
        c5 = max(in_channels[0] // 4, self.num_keypoints)
        self.branch_pose_detect = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(ch, c5, kernel_size=3, stride=1),
                ConvBNAct(c5, c5, kernel_size=3, stride=1),
                nn.Conv2d(c5, self.num_keypoints, 1)
            )
            for ch in in_channels
        )

    def decode_keypoints(self, batch_size, keypoints):
        ndim = self.keypoint_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:
                y = keypoints.view(batch_size, *self.keypoint_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                y = keypoints.view(batch_size, *self.keypoint_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(batch_size, self.num_keypoints, -1)
        else:
            y = keypoints.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid_()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y
    
    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchor_points=anchors, dim=1)
    
    def forward(self, x):
        batch_size = x[0].shape[0]
        angle = torch.cat([self.oriented_branch[i](x[i]).view(batch_size, self.num_extra, -1) for i in range(self.num_det_layers)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        keypoint = torch.cat([self.branch_pose_detect[i](x[i]).view(batch_size, self.num_keypoints, -1) 
                              for i in range(self.num_det_layers)], -1)

        if not self.training:
            self.angle = angle
        x = Detection.forward(self, x)
        if self.training:
            return x, keypoint, angle
        pred_keypoints = self.decode_keypoints(batch_size, keypoint)
        
        return torch.cat([x, pred_keypoints, angle], 1) if self.export else (torch.cat([x[0], pred_keypoints, angle], 1), (x[1], keypoint, angle))
    
class DEYODetection(Detection):
    def __init__(
        self,
        num_classes=80,
        in_channels=(512, 1024, 2048),
        nq=300,
        nh=8,  # num head
        ndl=2,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
    ):
        super().__init__(num_classes, in_channels)
        hd=in_channels[1]
        c3 = max(in_channels[0], min(self.num_classes, 100))  # channels
        self.branch_det.requires_grad = False
        self.num_queries = nq
        self.num_heads = nh
        self.num_decoder_layers = ndl
        self.feedforward_dims = d_ffn
        self.dropout_rate=dropout
        self.act = act
        self.eval_idx = eval_idx

        self.norm = nn.LayerNorm(hd)
        self.branch_cls = nn.ModuleList(nn.Sequential(nn.Sequential(ConvBNAct(x, c3, kernel_size=3, stride=1)), \
                                               nn.Sequential(ConvBNAct(c3, c3, kernel_size=3, stride=1)), \
                                                nn.Conv2d(c3, self.num_classes, 1)) for i, x in enumerate(in_channels))
        self.input_proj = nn.ModuleList(nn.Linear(x, hd) for x in in_channels)
        decoder_layer = ScaleAdaptiveDecoderLayer(hd, nh, d_ffn, dropout, act)
        self.decoder = ScaleAdaptiveTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, num_classes) for _ in range(ndl)])
        self._reset_parameters()
        
    def get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        feats = [feat.flatten(2).permute(0, 2, 1) for feat in x]
        for i, feat in enumerate(feats):
            feats[i] = self.input_proj[i](feat)
        feats = torch.cat(feats, 1)
        return feats
    
    def generate_anchors(self, x):
        for i in range(self.num_det_layers):
            x[i] = torch.cat((self.branch_det[i](x[i]), self.branch_cls[i](x[i])), 1)
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.num_outputs_per_anchor, -1) for xi in x], 2)
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in generate_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        dbox = self.decode_bboxes(self.distributed_focal_loss(box), self.anchors.unsqueeze(0)) * self.strides
        return x, dbox, cls
        
    def get_decoder_output(self, feats, dbox, cls, img_size):
        dbox = dbox.permute(0, 2, 1).contiguous().detach()
        cls = cls.permute(0, 2, 1).contiguous()
        
        bs = feats.shape[0]
        topk_ind = torch.topk(cls.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        embed = feats[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embed = self.norm(embed)
        
        dec_bboxes = dbox[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        refer_bbox = dec_bboxes / torch.tensor(img_size, device=dbox.device)[[1, 0, 1, 0]]
        
        dec_scores = self.decoder(
            embed,
            refer_bbox,
            self.dec_score_head,
            self.query_pos_head,
        )
        return dec_bboxes, dec_scores, topk_ind
    
    def forward(self, x, img_size=None):
        if self.stride[0] == 0:
            return super().forward(x)
        feats = self.get_encoder_input(x)
        preds, dbox, cls = self.generate_anchors(x)
        dec_bboxes, dec_scores, topk_ind = self.get_decoder_output(feats, dbox, cls, img_size)
        x = preds, dec_scores, topk_ind
        if self.training:
            return x
        y = torch.cat((dec_bboxes, dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)
        
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        bias_cls = bias_init_with_prob(0.01) / 80 * self.num_classes
        for cls_ in self.dec_score_head:
            constant_(cls_.bias, bias_cls)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer.weight)

    def get_module_info(self):
        return "DEYODetection", f"[{self.num_classes}, {self.in_channels}, {self.num_queries}, {self.num_heads}, {self.num_decoder_layers}, {self.feedforward_dims}, {self.dropout_rate}, {self.act}, {self.eval_idx}]"
    
class DFINETransformer(nn.Module):
    export = False

    def __init__(
        self,
        num_classes=80,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation=nn.ReLU(),
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
    ):
        super(DFINETransformer, self).__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_channels = feat_channels
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete")

        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.Tensor([0.5]), requires_grad = False)
        self.reg_scale = nn.Parameter(torch.Tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method
        )

        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method
        )

        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=mlp_act
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.target_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, mlp_act)
        
        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    (
                        "norm",
                        nn.LayerNorm(
                            hidden_dim,
                        ),
                    ),
                ]
            )
        )

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, mlp_act)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, mlp_act)

        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, mlp_act) for _ in range(self.eval_idx + 1)]
            + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, mlp_act) for _ in range(num_layers - self.eval_idx - 1)]
        )

        self.integral = Integral(self.reg_max)

        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_bbox_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)

            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.target_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()

        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                (
                                    "norm",
                                    nn.BatchNorm2d(
                                        self.hidden_dim
                                    ),
                                ),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes
    
    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []

        for level, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            level_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h*w, 4)
            anchors.append(level_anchors)
        
        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))

        return anchors, valid_mask
    
    def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.target_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
        
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list
    
    def _select_topk(self, memory: torch.Tensor, output_logits: torch.Tensor, output_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(output_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(output_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(output_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = output_anchors_unact.gather(
            dim=1, index = topk_ind.unsqueeze(-1).repeat(1, 1, output_anchors_unact.shape[-1])
        )

        topk_logits = ( output_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_logits.shape[-1])
            ) if self.training else None
        )

        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors
    
    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed.weight,
                self.num_denoising,
                self.label_noise_ratio,
                box_noise_scale=1.0,
                training=self.training
            )

        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale
            }
        
        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1]
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1]
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        if self.training:
            return out
        
        y = torch.cat((out_bboxes[-1], out_logits[-1]), -1)

        return y if self.export else (y, out)
    
    @torch.jit.unused
    def _set_aux_loss(self, output_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(output_class, outputs_coord)]
    
    @torch.jit.unused
    def _set_aux_loss2(
        self, outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None
    ):
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ]
    
    def get_module_info(self):
        return "DFINETransformer", f"[{self.num_classes}, {self.num_queries}, {self.hidden_dim}, {self.feat_channels}, {self.feat_strides}, {self.hidden_dim}, {self.num_levels}, {self.nhead}, {self.num_layers}]"
    
class LRPCHead(nn.Module):
    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv: nn.Conv2d) -> nn.Linear:
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear
    
    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool 
            )
        
class PEDetection(Detection):
    is_fused = False

    def __init__(self, num_classes = 80, embed_dim = 512, with_bn = False, in_channels = []):
        super().__init__(num_classes, in_channels)
        c3 = max(in_channels[0], min(self.num_classes, 100))
        assert c3 <= embed_dim
        assert with_bn
        self.branch_cls = (
            nn.ModuleList(nn.Sequential(ConvBNAct(ch, c3, 1, 3), ConvBNAct(c3, c3, 1, 3), nn.Conv2d(c3, embed_dim, 1)) for ch in in_channels) 
            if self.legacy
            else
            nn.ModuleList(nn.Sequential(nn.Sequential(DepthwiseConvBNAct(ch, ch, 1, 3), ConvBNAct(ch, c3, 1, 1)),
                                        nn.Sequential(DepthwiseConvBNAct(c3, c3, 1, 3), ConvBNAct(c3, c3, 1, 1)),
                                        nn.Conv2d(c3, embed_dim, 1)
                                        )
                            for ch in in_channels
            )
        )

        self.branch3 = nn.ModuleList(BNContrastiveHead(embed_dims=embed_dim) if with_bn else ContrastiveHead() for _ in in_channels)

        self.reprta = Residual(SwiGLUFFN2(embed_dim, embed_dim))
        self.savpe = SAVPE(in_channels, c3, embed_dim)
        self.embed = embed_dim

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        if self.is_fused:
            return
        
        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.branch_cls, self.branch3):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm
            t = txt_feats * logit_scale.exp()

            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                ).requires_grad_(False).to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv

            bn_head.fuse()
        
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)
    
    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        if vpe.shape[1] == 0:
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3
        return vpe
    
    def forward_lrpc(self, x: list[torch.Tensor], return_mask: bool = False) -> torch.Tensor | tuple:
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.num_det_layers):
            cls_feat = self.branch_cls[i](x[i])
            loc_feat = self.branch_det[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](
                cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
            )
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in generate_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.distributed_focal_loss(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.distributed_focal_loss(box), self.anchors.unsqueeze(0)) * self.strides

        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)

        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)
        
    def forward(self, x: list[torch.Tensor], cls_pe: torch.Tensor, return_mask: bool = False) -> torch.Tensor | tuple:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x, return_mask)
        for i in range(self.num_det_layers):
            x[i] = torch.cat((self.branch_det[i](x[i]), self.branch3[i](self.branch_cls[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.num_outputs_per_anchor = self.num_classes + self.reg_max * 4

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.num_outputs_per_anchor, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in generate_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.distributed_focal_loss(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.distributed_focal_loss(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
    
    def bias_init(self):
        m = self
        for a, b, c, s in zip(m.branch_det, m.branch_cls, m.branch3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.num_classes / (640 / s) ** 2)

    def get_module_info(self):
        return f"PEDetection", f"[{self.num_classes}, {self.in_channels}, {self.embed}]"


class PESegmentation(PEDetection):
    def __init__(self, num_classes=80, num_masks=32, num_protos=256, embed_dim=512, with_bn=False, in_channels=[]):
        super().__init__(num_classes, embed_dim, with_bn, in_channels)
        self.num_masks = num_masks
        self.num_protos = num_protos
        self.proto = ProtoMaskModule(in_channels[0], self.num_protos, self.num_masks)

        c5 = max(in_channels[0] // 4, self.num_masks)
        self.branch4 = nn.ModuleList(nn.Sequential(ConvBNAct(x, c5, 1, 3), ConvBNAct(c5, c5, 1, 3), nn.Conv2d(c5, self.num_masks, 1)) for x in in_channels)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> tuple | torch.Tensor:
        proto_masks = self.proto(x[0])
        bs = proto_masks.shape[0]

        mask_coefficients = torch.cat([self.branch4[i](x[i]).view(bs, self.num_masks, -1) for i in range(self.num_det_layers)], 2)
        has_lrpc = hasattr(self, "lrpc")

        if not has_lrpc:
            x = PEDetection.forward(self, x, text)
        else:
            x, mask = PEDetection.forward(self, x, text, return_mask=True)

        if self.training:
            return x, mask_coefficients, proto_masks
        
        if has_lrpc:
            mask_coefficients = (mask_coefficients * mask.int()) if self.export and not self.dynamic else mask_coefficients[..., mask]
        
        return (torch.cat([x, mask_coefficients], 1), proto_masks) if self.export else (torch.cat([x[0], mask_coefficients], 1), (x[1], mask_coefficients, proto_masks))
    
    def get_module_info(self):
        return f"PESegmentation", f"[{self.num_classes}, {self.num_masks}, {self.num_protos}, {self.in_channels}, {self.embed}]"
# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_ 
from vajra.tal.anchor_generator import dist2bbox, dist2rbox, generate_anchors
from vajra.nn.modules import DepthwiseConvBNAct, ConvBNAct, DistributedFocalLoss, ProtoMaskModule, UConv, BNContrastiveHead, ContrastiveHead, ImagePoolingAttention
from vajra.nn.transformer import ScaleAdaptiveDecoderLayer, ScaleAdaptiveTransformerDecoder, MLP
from vajra.utils import LOGGER
from vajra.nn.utils import bias_init_with_prob

class Classification(nn.Module):
    def __init__(self, in_c, out_c, hidden_c=2048, kernel_size=1, stride=1, padding=None, groups=1) -> None:
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
        return x if self.training else x.softmax(1)

    def get_module_info(self):
        return f"Classification", f"[{self.in_c}, {self.out_c}, {self.hidden_c}, {self.kernel_size}, {self.stride}, {self.padding}, {self.groups}]"

class Detection(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

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
        self.branch_cls = nn.ModuleList(
            nn.Sequential(nn.Sequential(DepthwiseConvBNAct(ch, ch, 1, 3), ConvBNAct(ch, c3, 1, 1)),
                          nn.Sequential(DepthwiseConvBNAct(c3, c3, 1, 3), ConvBNAct(c3, c3, 1, 1)),
                          nn.Conv2d(c3, self.num_classes, 1))
                          for ch in in_channels
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
        if self.export:
            return dist2bbox(bboxes, anchor_points=anchors, xywh=False, dim=1)
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

class Segementation(Detection):
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
                y[:, 2::3] = y[:, 2::3].sigmoid_()
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
        return dist2rbox(bboxes, self.angle, anchor_points=anchors, xywh=True, dim=1)

    def get_module_info(self):
        return f"OBBDetection", f"[{self.num_classes}, {self.num_extra}, {self.in_channels}]"

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
        
    def get_decoder_output(self, feats, dbox, cls, imgsz):
        dbox = dbox.permute(0, 2, 1).contiguous().detach()
        cls = cls.permute(0, 2, 1).contiguous()
        
        bs = feats.shape[0]
        topk_ind = torch.topk(cls.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        embed = feats[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embed = self.norm(embed)
        
        dec_bboxes = dbox[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        refer_bbox = dec_bboxes / torch.tensor(imgsz, device=dbox.device)[[1, 0, 1, 0]]
        
        dec_scores = self.decoder(
            embed,
            refer_bbox,
            self.dec_score_head,
            self.query_pos_head,
        )
        return dec_bboxes, dec_scores, topk_ind
    
    def forward(self, x, imgsz=None):
        if self.stride[0] == 0:
            return super().forward(x)
        feats = self.get_encoder_input(x)
        preds, dbox, cls = self.generate_anchors(x)
        dec_bboxes, dec_scores, topk_ind = self.get_decoder_output(feats, dbox, cls, imgsz)
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
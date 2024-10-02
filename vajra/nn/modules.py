# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Tuple, Optional, List, Union, Any, Callable

act_table = {'silu' : nn.SiLU(),
             'relu' : nn.ReLU(),
             'relu6': nn.ReLU6(),
             'hardswish': nn.Hardswish(),
             'mish': nn.Mish()}

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

class Conv(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=1, padding=None, groups=1, dilation=1, bias=False) -> None:
        super().__init__()
        if padding == None:
            padding = kernel_size // 2

        if dilation > 1:
            kernel_size = dilation * (kernel_size - 1) + 1
        
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        self.conv = nn.Conv2d(in_c, 
                              out_c, 
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
    
    def forward(self, x):
        return self.conv(x)

    def get_module_info(self):
        return "Conv", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.padding}, {self.groups}, {self.dilation}, {self.bias}]"

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        return self.linear(x)

    def get_module_info(self):
        return "Linear", f"[{self.in_features}, {self.out_features}, {self.bias}]"

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=1, padding=None, groups=1, dilation=1, bias=False , act='silu') -> None:
        super().__init__()
        if padding==None:
            padding = kernel_size // 2
        if dilation > 1:
            kernel_size = dilation * (kernel_size - 1) + 1
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.act = act
        self.conv = nn.Conv2d(in_c, 
                              out_c, 
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act_table.get(act) if act is not None else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def get_module_info(self):
        return f"ConvBNAct", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.padding}, {self.act}]"

class DepthwiseConvBNAct(ConvBNAct):
    def __init__(self, in_c, out_c, stride=1, kernel_size=1, padding=None, groups=1, dilation=1, bias=False, act='silu') -> None:
        super().__init__(in_c, out_c, stride, kernel_size, padding, groups=math.gcd(in_c, out_c), dilation=dilation, bias=bias, act=act)

class DepthWiseSeparableConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=1, padding=None, groups=1, dilation=1, bias=False, act='silu') -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        self.act = act
        self.dwconv = DepthwiseConvBNAct(in_c, in_c, stride, kernel_size, padding, groups, dilation, bias, act)
        self.pwconv = ConvBNAct(in_c, out_c, 1, 1, 0, act=act)

    def forward(self, x):
        dw = self.dwconv(x)
        pw = self.pwconv(dw)
        return pw

    def get_module_info(self):
        return "DepthwiseSeparableConvBNAct", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.padding}, {self.groups}, {self.dilation}, {self.bias}, {self.act}]"

class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0) -> None:
        super().__init__()
        self.kernel_size=kernel_size
        self.stride=stride     
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)

    def get_module_info(self):
        return f"MaxPool2d", f"[{self.kernel_size}, {self.stride}]"

class AdaptiveAvgPool2D(nn.Module):
    def __init__(self, out_size) -> None:
        super().__init__()
        self.out_size = out_size
        self.pool = nn.AdaptiveAvgPool2d(out_size)
    
    def forward(self, x):
        return self.pool(x)

    def get_module_info(self):
        return "AdaptiveAvgPool2D", f"[{self.out_size}]"

class PyConv(nn.Module):
    def __init__(self, in_c, out_c, pyconv_kernels=[3, 5, 7, 9], shortcut=False):
        super(PyConv, self).__init__()
        assert in_c == out_c
        self.conv1_1 = ConvBNAct(in_c, in_c//4, k=pyconv_kernels[0])
        self.conv1_2 = ConvBNAct(in_c, in_c//4, k=pyconv_kernels[1])
        self.conv1_3 = ConvBNAct(in_c, in_c//4, k=pyconv_kernels[2])
        self.conv1_4 = ConvBNAct(in_c, in_c//4, k=pyconv_kernels[3])
        self.add = shortcut
        #self.conv2 = Conv(c1, c2, 1, 1)

    def forward(self, x):
        pyramid = torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x), self.conv1_4(x)), dim=1)
        return x + pyramid if self.add else pyramid

class PyConv3(nn.Module):
    """ Pyramidal Convolutional Block with 3 convolutions """

    def __init__(self, in_c, out_c,  pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8], shortcut=False):
        super(PyConv3, self).__init__()
        assert in_c == out_c
        self.conv1_1 = ConvBNAct(in_c, in_c//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=1, groups=pyconv_groups[0])
        self.conv1_2 = ConvBNAct(in_c, in_c//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=1, groups=pyconv_groups[1])
        self.conv1_3 = ConvBNAct(in_c, in_c//2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=1, groups=pyconv_groups[2])
        self.add = shortcut

    def forward(self, x):
        pyramid = torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x)), dim=1)
        return x + pyramid if self.add else pyramid

class PyConv2(nn.Module):
    """ Pyramidal Convolutional Block with 2 convolutions and groups = 1 """
    def __init__(self, in_c, out_c,  pyconv_kernels=[3, 5], shortcut=False):
        super(PyConv2, self).__init__()
        assert in_c == out_c
        self.conv1_1 = ConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[0])
        self.conv1_2 = ConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[1])
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        pyramid = torch.cat((self.conv1_1(x), self.conv1_2(x)), dim=1)
        return x + pyramid if self.add else pyramid

class DWPyConv2(nn.Module):
    """ Pyramidal Convolutional Block with 2 convolutions and groups = 1 """
    def __init__(self, in_c, out_c,  pyconv_kernels=[3, 5], shortcut=False):
        super(DWPyConv2, self).__init__()
        assert in_c == out_c
        self.conv1_1 = DepthwiseConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[0])
        self.conv1_2 = DepthwiseConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[1])
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        pyramid = torch.cat((self.conv1_1(x), self.conv1_2(x)), dim=1)
        return x + pyramid if self.add else pyramid

class DWPyConv3(nn.Module):
    """ Pyramidal Convolutional Block with 3 convolutions and groups = 1 """
    def __init__(self, in_c, out_c,  pyconv_kernels=[3, 5, 7], shortcut=False):
        super(DWPyConv3, self).__init__()
        assert in_c == out_c
        self.conv1 = DepthwiseConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[0])
        self.conv2 = DepthwiseConvBNAct(in_c, out_c//4, kernel_size=pyconv_kernels[1])
        self.conv3 = DepthwiseConvBNAct(in_c, out_c//4, kernel_size=pyconv_kernels[2])
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        pyramid = torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), dim=1)
        return x + pyramid if self.add else pyramid

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, in_c, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(in_c)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))

class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    """Standard bottleneck"""

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3,3), expansion_ratio=0.5, groups=1, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class BottleneckResNet(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample.get_module_info() if downsample else downsample
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation

        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = ConvBNAct(inplanes, width, 1, 1, act="relu")
        self.conv2 = ConvBNAct(width, width, stride, kernel_size=3, groups=groups, dilation=dilation, act="relu")
        self.conv3 = ConvBNAct(width, planes * self.expansion, 1, 1, act=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def get_module_info(self):
        return "BottleneckResNet", f"[{self.inplanes}, {self.planes}, {self.stride}, {self.downsample}, {self.groups}, {self.base_width}, {self.dilation}]"

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample.get_module_info()[0] if downsample else downsample
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = ConvBNAct(inplanes, planes, stride, kernel_size=3, act="relu")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBNAct(planes, planes, stride=1, kernel_size=3, act=None)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def get_module_info(self):
        return "BasicBlock", f"[{self.inplanes}, {self.planes}, {self.stride}, {self.downsample}, {self.groups}, {self.base_width}, {self.dilation}]"

class BottleneckV2(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3,3), expansion_ratio=0.5, groups=1):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=None)
        self.act = nn.SiLU()
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x))) if self.add else self.act(self.conv2(self.conv1(x)))

class BottleneckPyConv(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3, 3), expansion_ratio=0.5, groups=1):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1)
        self.pyconv = PyConv2(hidden_c, hidden_c, shortcut=shortcut)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size[1], 1, groups)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.pyconv(self.conv1(x))) if self.add else self.conv2(self.pyconv(self.conv1(x)))

class BottleneckDWConv(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3, 3), expansion_ratio=0.5, groups=1):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1)
        self.dwconv = DepthwiseConvBNAct(hidden_c, hidden_c, stride=1, kernel_size=5)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size[1], 1, groups)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.dwconv(self.conv1(x))) if self.add else self.conv2(self.dwconv(self.conv1(x)))

class VajraPyConvBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, num_pyconv=3) -> None:
        super().__init__()
        if num_pyconv == 2:
            block = PyConv2
        elif num_pyconv == 3:
            block = PyConv3
        else:
            block = PyConv
        hidden_c = int(out_c * 0.5)
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=1)
        self.pyconv_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut) for _ in num_blocks)
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1)*hidden_c, out_c, kernel_size=1)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(pyconv(y[-1]) for pyconv in self.pyconv_blocks)
        y = self.conv2(torch.cat(y, 1))
        return y + x if self.add else y

class PoolingAttention(nn.Module):
    def __init__(self, in_channels=(), ec=256, ct=512, num_heads=8, kernel_size=3, scale=False) -> None:
        super().__init__()
        num_features = len(in_channels)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(ch, ec, kernel_size=1) for ch in in_channels])
        self.img_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((kernel_size, kernel_size)) for _ in range(num_features)])
        self.ec = ec
        self.num_heads = num_heads
        self.num_features = num_features
        self.head_channels = ec // num_heads
        self.kernel_size = kernel_size
    
    def forward(self, x):
        batch_size = x[0].shape[0]
        assert len(x) == self.num_features
        num_patches = self.kernel_size ** 2
        x = [pool(proj(x)).view(batch_size, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.img_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(x.mean(dim=1, keepdim=True))
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(batch_size, -1, self.num_heads, self.head_channels)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_channels)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_channels)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.head_channels**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(batch_size, -1, self.ec))
        return x * self.scale

class VajraBottleneckBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, bottleneck_pyconv=False) -> None:
        super().__init__()
        block = BottleneckPyConv if bottleneck_pyconv else Bottleneck
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.bottleneck_pyconv = bottleneck_pyconv
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return f"VajraBottleneckBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_pyconv}]"

class VajraBottleneckAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, embed_channels=128, num_heads=1, guide_channels=512) -> None:
        super().__init__()
        block = Bottleneck
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.guide_channels = guide_channels

        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c)
        self.attn = MaxSigmoidAttentionBlock(hidden_c, hidden_c, num_heads, embed_channels, guide_channels)

    def forward(self, x, guide):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y.append(self.attn(y[-1], guide))
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return f"VajraBottleneckAttentionBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.embed_channels}, {self.num_heads}, {self.guide_channels}]"

class VajraV2BottleneckBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1,bottleneck_pyconv=False) -> None:
        super().__init__()
        block = VajraBottleneckBlock
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.bottleneck_pyconv = bottleneck_pyconv
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, num_blocks=1, shortcut=shortcut, kernel_size=1) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return f"VajraV2BottleneckBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_pyconv}]"

class VajraPyConvBottleneckBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False) -> None:
        super().__init__()
        mid_c = out_c // 2
        self.add = shortcut and in_c == out_c
        self.conv1 = ConvBNAct(in_c, mid_c, kernel_size=1)
        self.pyconv = PyConv2(mid_c, mid_c, shortcut=shortcut)
        self.bottleneck_blocks = nn.ModuleList(Bottleneck(mid_c, mid_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks+2) * mid_c, out_c, k=1, s=1)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y = y + [self.pyconv(y[-1])]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y,1))
        return y + x if self.add else y

class VajraStemBlock(nn.Module):
    """Vajra Stem Block with 4 convolutions (2 Convolutions and 1 Residual Bottleneck with 2 convolutions) and 1 MaxPool"""

    def __init__(self, in_c, hidden_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=3, stride=2)
        self.bottleneck = Bottleneck(hidden_c, hidden_c, True)
        self.conv2 = ConvBNAct(hidden_c * 2, hidden_c, kernel_size=3, stride=2)
        self.conv3 = ConvBNAct(hidden_c, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        stem = self.conv1(x)
        branch1 = self.bottleneck(stem)
        join_branches = torch.cat([stem, branch1], dim=1)
        downsample = self.conv2(join_branches)
        out = self.conv3(downsample)
        return out

    def get_module_info(self):
        return f"VajraStemBlock", f"[{self.in_c}, {self.hidden_c}, {self.out_c}]"

class VajraV2StemBlock(nn.Module):
    def __init__(self, in_c, hidden_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=3, stride=2)
        self.vajra_bottleneck = VajraBottleneckBlock(in_c=hidden_c, out_c=hidden_c, num_blocks=2, shortcut=True)
        self.conv2 = ConvBNAct(hidden_c * 2, hidden_c, kernel_size=3, stride=2)
        self.conv3 = ConvBNAct(hidden_c, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        stem = self.conv1(x)
        branch1 = self.vajra_bottleneck(stem)
        join_branches = torch.cat([stem, branch1], dim=1)
        downsample = self.conv2(join_branches)
        out = self.conv3(downsample)
        return out

    def get_module_info(self):
        return f"VajraV2StemBlock", f"[{self.in_c}, {self.hidden_c}, {self.out_c}]"

class YOLOBottleneckBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, reduction_ratio=0.5) -> None:
        super().__init__()
        self.hidden = int(out_c * reduction_ratio)
        self.conv1 = ConvBNAct(in_c, 2 * self.hidden, 1, 1)
        self.conv2 = ConvBNAct((2 + num_blocks) * self.hidden, out_c, 1)
        self.bottleneck_blocks = nn.ModuleList(Bottleneck(self.hidden, self.hidden, shortcut=shortcut, kernel_size=(3, 3), reduction_ratio=0.5))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.conv1(x).split((self.hidden, self.hidden), 1))
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

class MaxSigmoidAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=1, embed_channels=512, guid_channels=512, scale=False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_channels = out_c // num_heads
        self.embed_conv = ConvBNAct(in_c, out_c, kernel_size=1, act=None) if in_c != embed_channels else None
        self.guide_linear = nn.Linear(guid_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.proj_conv = ConvBNAct(in_c, out_c, kernel_size=3, stride=1, act=None)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        batch, _, h, w = x.shape

        guide = self.guide_linear(guide)
        guide = guide.view(batch, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.view(batch, self.num_heads, self.head_channels, h, w)

        attn_weights = torch.einsum("bmchw, bnmc->bmhwn", embed, guide)
        attn_weights = attn_weights.max(dim=-1)[0]
        attn_weights = attn_weights / (self.head_channels**0.5)
        attn_weights = attn_weights + self.bias[None, :, None, None]
        attn_weights = attn_weights.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(batch, self.num_heads, -1, h, w)
        x = x * attn_weights.unsqueeze(2)
        return x.view(batch, -1, h, w)

class SEBlock(nn.Module):
    """ Code Based on https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py """

    def __init__(self, in_c, out_c, reduction=2) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels = in_c,
                               out_channels = out_c // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels = out_c // reduction,
                               out_channels = out_c,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.act2 = nn.Hardsigmoid()
    
    def forward(self, x):
        residual = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        out = residual + x
        return out

class Fusion3CBAM(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        total_c = sum(in_c)
        self.conv_fused = ConvBNAct(total_c, out_c, 1, 1)
        self.cbam = CBAM(out_c)
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        _, _, H, W = x[1].shape
        output_size = [int(H), int(W)]

        x0 = self.downsample(x[0], output_size)
        x1 = x[1]
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        concatenated_oup = torch.cat((x0, x1, x2), dim=1)
        fused = self.conv_fused(concatenated_oup)
        cbam = self.cbam(fused)
        return fused + cbam

class Fusion4CBAM(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        total_c = sum(in_c)
        self.conv_fused = ConvBNAct(total_c, self.out_c, 1, 1)
        self.cbam = CBAM(out_c)

    def forward(self, x):
        _, _, H, W = x[2].shape
        x0 = F.interpolate(x[0], size=(H, W), mode="nearest") #adaptive_pool(x[0])
        x1 = F.interpolate(x[1], size=(H, W), mode="nearest") #adaptive_pool(x[1])
        x2 = x[2]
        x3 = F.interpolate(x[3], size=(H, W), mode='nearest')
        concatenated_oup = torch.cat((x0, x1, x2, x3), dim=1)
        fused = self.conv_fused(concatenated_oup)
        cbam = self.cbam(fused)
        return fused + cbam

    def get_module_info(self):
        return f"Fusion4CBAM", f"[{self.in_c}, {self.out_c}]"

class PyramidalPoolCBAM(nn.Module):
    def __init__(self, in_c, out_c, stride=2, use_cbam=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        total_c = sum(in_c)
        self.use_cbam = use_cbam
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.conv_fused = ConvBNAct(total_c, out_c, 1, 1)
    
    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        out = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs] #[adaptive_pool(inp) for inp in inputs]
        concatenated_out = torch.cat(out, dim=1)
        conv = self.conv_fused(concatenated_out)
        cbam = self.cbam(conv)
        return cbam + conv if self.use_cbam else conv

    def get_module_info(self):
        return f"PyramidalPoolCBAM", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.use_cbam}]"

class VajraDownsample(nn.Module):
    """ Inspired by the Stem Block of PPHGNetV2
        Performs 5 convolutions and 2 maxpool operations for stride of 2
    """
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.stem1 = ConvBNAct(in_c, mid_c, 3, 2, act=nn.ReLU())
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, 2, 1, 0, act=nn.ReLU())
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, 3, 1, act=nn.ReLU())
        self.stem4 = ConvBNAct(mid_c, out_c, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.pool(x)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class VajraSPPModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        hidden_c = in_c // 2
        self.cv1 = ConvBNAct(in_c, hidden_c, 3, 1)
        self.cv2 = ConvBNAct(hidden_c * 4, out_c, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.cbam = CBAM(out_c)

    def forward(self, x):
        y = self.cv1(x)
        y1 = self.m1(y)
        y2 = self.m2(y1)
        y3 = self.m3(y2)
        z = self.cv2(torch.cat((x, y, y1, y2, y3), 1))
        final = self.cbam(z)
        return final + z

class SPPF(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        hidden_dim = in_c // 2
        self.conv1 = ConvBNAct(in_c, hidden_dim, 1, 1)
        self.conv2 = ConvBNAct(hidden_dim * 4, out_c, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.maxpool(y2)), 1))

class MBConvEffNet(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion_ratio=4, kernel_size=3) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.expansion_ratio = expansion_ratio
        self.hidden_dim = round(self.in_c * self.expansion_ratio)
        self.add = self.stride == 1 and self.in_c == self.out_c
        self.kernel_size = kernel_size
        
        self.padding = (self.kernel_size - 1) // 2
        self.block = nn.Sequential(
                ConvBNAct(self.in_c, self.hidden_dim, 1, 1, 0),
                DepthwiseConvBNAct(self.hidden_dim, self.hidden_dim, stride=self.stride, kernel_size=self.kernel_size, padding=self.padding, groups=self.hidden_dim),
                SELayer(self.hidden_dim, reduction=self.expansion_ratio*4),
                ConvBNAct(self.hidden_dim, self.out_c, 1, 1, 0)
            )

    def forward(self, x):
        return x + self.block(x) if self.add else self.block(x)

    def get_module_info(self):
        return "MBConvEffNet", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.expansion_ratio}, {self.kernel_size}]"


class FusedMBConvEffNet(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion_ratio=4, use_se=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.expansion_ratio = expansion_ratio
        self.hidden_dim = round(self.in_c * self.expansion_ratio)
        self.use_se = use_se
        self.add = self.stride == 1 and self.in_c == self.out_c

        if not self.use_se:
            self.block = nn.Sequential(
                ConvBNAct(self.in_c, self.hidden_dim, stride=self.stride, kernel_size=3, padding=1),
                ConvBNAct(self.hidden_dim, self.out_c, 1, 1, act=None)
            )
        else:
            self.block = nn.Sequential(
                ConvBNAct(self.in_c, self.hidden_dim, stride=self.stride, kernel_size=3, padding=1),
                SELayer(self.hidden_dim, reduction=self.expansion_ratio*4),
                ConvBNAct(self.hidden_dim, self.out_c, 1, 1, act=None)
            )
    
    def forward(self, x):
        return x + self.block(x) if self.add else self.block(x)

    def get_module_info(self):
        return "FusedMBConvEffNet", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.expansion_ratio}, {self.use_se}]"


class VajraMBConvBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, fused=True, use_se=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.add = shortcut and in_c == out_c
        self.use_se = use_se
        self.fused = fused
        block = FusedMBConvEffNet if self.fused else MBConvEffNet
        hidden_c = int(out_c * 0.5)
        self.shortcut = shortcut
        self.kernel_size = kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        if block == FusedMBConvEffNet:
            self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, stride=1, expansion_ratio=1, use_se=self.use_se) for _ in range(num_blocks))
        else:
            self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, stride=1, expansion_ratio=1) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return "VajraMBConvBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.fused}, {self.use_se}]"

class VajraWindowAttnBottleneck(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, window_size=7, num_heads=3) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.add = shortcut and in_c == out_c
        block = BottleneckAttn
        hidden_c = int(out_c * 0.5)
        self.shortcut = shortcut
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        #self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks))
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, window_size=window_size, num_heads=num_heads, expansion_ratio=1.0) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        #B, C, H, W = y.shape
        #attn = y.view(B, C, H * W).transpose(1, 2)  # Reshape to [B, N, C], where N = H * W
        #attn = self.window_attn(attn)  # Pass the reshaped tensor to attention
        #attn = attn.view(B, H, W, C).permute(0, 3, 1, 2)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return "VajraWindowAttnBottleneck", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.window_size}, {self.num_heads}]"

class VajraConvNeXtBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.add = shortcut and in_c == out_c
        block = VajraV2Block #ConvNeXtV2Block
        hidden_c = int(out_c * 0.5)
        self.shortcut = shortcut
        self.kernel_size = kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(dim=hidden_c) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return "VajraConvNeXtBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}]"


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def get_module_info(self):
        return f"LayerNorm", f"[{self.normalized_shape}, {self.eps}, {self.data_format}]"

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class MixConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(1, 3), stride=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(kernel_size)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, out_c).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [out_c] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(in_c, int(c_), k, stride, k // 2, groups=math.gcd(in_c, int(c_)), bias=False) for k, c_ in zip(kernel_size, c_)])
        #self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU()

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)

class MixConvNeXtEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., expand_ratio=4, dwconv_kernels=[3,5]):
        super().__init__()
        self.dim = dim
        self.drop_path_ratio = drop_path
        self.expand_ratio = expand_ratio
        self.dwconv_kernels = dwconv_kernels

        self.mixconv = MixConv2d(dim,dim,k=dwconv_kernels)#BiDWConv(dim, dim, dwconv_kernels)#nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expand_ratio * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.Hardswish()
        self.grn = GRN(expand_ratio * dim)
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.mixconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def get_module_info(self):
        return "MixConvNeXtEncoder", f"[{self.dim}, {self.drop_path_ratio}, {self.expand_ratio}, {self.dwconv_kernels}]"

class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos

    def get_module_info(self):
        return "PositionalEncodingFourier", f"[{self.hidden_dim}, {self.dim}, {self.temperature}]"

class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_drop
        self.proj_dropout = proj_drop
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def get_module_info(self):
        return "XCA", f"[{self.dim}, {self.num_heads}, {self.qkv_bias}, {self.attn_dropout}, {self.proj_dropout}]"

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class SDTAEncoderBNHS(nn.Module):
    """
        SDTA Encoder with Batch Norm and Hard-Swish Activation
        Taken from - EdgNeXt : Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1):
        super().__init__()
        self.dim = dim
        self.drop_path_ratio = drop_path
        self.layer_scale_init_value = layer_scale_init_value
        self.expand_ratio = expan_ratio
        self.use_pos_embed = use_pos_emb
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_drop
        self.drop_rate = drop
        self.scales = scales

        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = nn.BatchNorm2d(dim)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.Hardswish()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        # XCA
        x = self.norm_xca(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Inverted Bottleneck
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

    def get_module_info(self):
        return "SDTAEncoderBNHS", f"[{self.dim}, {self.drop_path_ratio}, {self.layer_scale_init_value}, {self.expand_ratio}, {self.use_pos_embed}, {self.num_heads}, {self.qkv_bias}, {self.attn_dropout}, {self.drop_rate}, {self.scales}]"

class SDTAEncoderBNHS_GRN(nn.Module):
    """
        SDTA Encoder with Batch Norm and Hard-Swish Activation
        Taken from - EdgNeXt : Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1):
        super().__init__()
        self.dim = dim
        self.drop_path_ratio = drop_path
        self.layer_scale_init_value = layer_scale_init_value
        self.expand_ratio = expan_ratio
        self.use_pos_embed = use_pos_emb
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_drop
        self.drop_rate = drop
        self.scales = scales

        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = nn.BatchNorm2d(dim)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.Hardswish()  # TODO: MobileViT is using 'swish'
        self.grn = GRN(expan_ratio * dim)
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        #self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  #requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        # XCA
        x = self.norm_xca(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Inverted Bottleneck
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        #if self.gamma is not None:
            #x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

    def get_module_info(self):
        return "SDTAEncoderBNHS_GRN", f"[{self.dim}, {self.drop_path_ratio}, {self.layer_scale_init_value}, {self.expand_ratio}, {self.use_pos_embed}, {self.num_heads}, {self.qkv_bias}, {self.attn_dropout}, {self.drop_rate}, {self.scales}]"

class ConvEncoder_BNHS_EdgeNeXt(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dim = dim
        self.drop_path_ratio = drop_path
        self.layer_scale_init_val = layer_scale_init_value
        self.expand_ratio = expan_ratio
        self.kernel_size = kernel_size

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def get_module_info(self):
        return "ConvEncoder_BNHS_EdgeNeXt", f"[{self.dim}, {self.drop_path_ratio}, {self.layer_scale_init_val}, {self.expand_ratio}, {self.kernel_size}]"

class ConvEncoder_BNHS_EdgeNeXt_GRN(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dim = dim
        self.drop_path_ratio = drop_path
        self.layer_scale_init_val = layer_scale_init_value
        self.expand_ratio = expan_ratio
        self.kernel_size = kernel_size

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.SiLU()
        self.grn = GRN(expan_ratio * dim)
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def get_module_info(self):
        return "ConvEncoder_BNHS_EdgeNeXt_GRN", f"[{self.dim}, {self.drop_path_ratio}, {self.layer_scale_init_val}, {self.expand_ratio}, {self.kernel_size}]"

class ConvNeXtV1Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.drop_path_prob = drop_path
        self.layer_scale_init_value = layer_scale_init_value
        self.dwconv = Conv(dim, dim, stride=1, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def get_module_info(self):
        return "ConvNeXtV1Block", f"[{self.dim}, {self.drop_path_prob}, {self.layer_scale_init_value}]"

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbedding(nn.Module):
    """
    Module embeds a given image into patch embeddings.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size = 4) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param patch_size: (int) Patch size to be utilized
        :param image_size: (int) Image size to be used
        """
        super(PatchEmbedding, self).__init__()
        self.out_channels: int = out_channels
        self.linear_embedding: nn.Module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=(patch_size, patch_size),
                                                     stride=(patch_size, patch_size))
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        embedding = self.linear_embedding(x)
        embedding = embedding.permute(0, 2, 3, 1)
        embedding = self.normalization(embedding)
        embedding = embedding.permute(0, 3, 1, 2)
        return embedding

def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    batch_size: int = int(input.shape[0] // (height * width // window_size // window_size))
    # Reshape input to
    output: torch.Tensor = input.view(batch_size, height // window_size, width // window_size, channels,
                                      window_size, window_size)
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output

def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)
    return output

class WindowMultiHeadAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, meta_hidden_feats = 256, sequential_self_attention = False, attn_drop=0., proj_drop=0.,) -> None:
        super().__init__()
        assert (dim % num_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sequential_self_attention: bool = sequential_self_attention
        self.mapping_qkv: nn.Module = nn.Linear(dim, dim * 3, bias=True)
        self.attention_dropout: nn.Module = nn.Dropout(attn_drop)
        self.projection: nn.Module = nn.Linear(dim, dim, bias=True)
        self.projection_dropout: nn.Module = nn.Dropout(proj_drop)
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_hidden_feats, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_hidden_feats, out_features=num_heads, bias=True)
        )
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, num_heads, 1, 1)))
        # Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()
    
    def __make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases
        """
        indexes: torch.Tensor = torch.arange(self.window_size, device=self.tau.device)
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes]), dim=0)
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1)
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log(1. + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log)

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param kwargs: (Any) Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias: torch.Tensor = self.meta_network(self.relative_coordinates_log)
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0)
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.num_heads,
                                                                              self.window_size * self.window_size,
                                                                              self.window_size * self.window_size)
        return relative_position_bias.unsqueeze(0)

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Compute attention map with scaled cosine attention
        attention_map: torch.Tensor = torch.einsum("bhqd, bhkd -> bhqk", query, key) \
                                      / torch.maximum(torch.norm(query, dim=-1, keepdim=True)
                                                      * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1),
                                                      torch.tensor(1e-06, device=query.device, dtype=query.dtype))
        attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)
        # Apply relative positional encodings
        attention_map: torch.Tensor = attention_map + self.__get_relative_positional_encodings()
        # Apply mask if utilized
        if mask is not None:
            number_of_windows: int = mask.shape[0]
            attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows, number_of_windows,
                                                             self.num_heads, tokens, tokens)
            attention_map: torch.Tensor = attention_map + mask.unsqueeze(1).unsqueeze(0)
            attention_map: torch.Tensor = attention_map.view(-1, self.num_heads, tokens, tokens)
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)
        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)
        # Apply attention map and reshape
        output: torch.Tensor = torch.einsum("bhal, bhlv -> bhav", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def __sequential_self_attention(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor,
                                    value: torch.Tensor,
                                    batch_size_windows: int,
                                    tokens: int,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        output: torch.Tensor = torch.ones_like(query)
        # Compute relative positional encodings fist
        relative_position_bias: torch.Tensor = self.__get_relative_positional_encodings()
        # Iterate over query and key tokens
        for token_index_query in range(tokens):
            # Compute attention map with scaled cosine attention
            attention_map: torch.Tensor = \
                torch.einsum("bhd, bhkd -> bhk", query[:, :, token_index_query], key) \
                / torch.maximum(torch.norm(query[:, :, token_index_query], dim=-1, keepdim=True)
                                * torch.norm(key, dim=-1, keepdim=False),
                                torch.tensor(1e-06, device=query.device, dtype=query.dtype))
            attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)[..., 0]
            # Apply positional encodings
            attention_map: torch.Tensor = attention_map + relative_position_bias[..., token_index_query, :]
            # Apply mask if utilized
            if mask is not None:
                number_of_windows: int = mask.shape[0]
                attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows,
                                                                 number_of_windows, self.num_heads, 1,
                                                                 tokens)
                attention_map: torch.Tensor = attention_map \
                                              + mask.unsqueeze(1).unsqueeze(0)[..., token_index_query, :].unsqueeze(3)
                attention_map: torch.Tensor = attention_map.view(-1, self.num_heads, tokens)
            attention_map: torch.Tensor = attention_map.softmax(dim=-1)
            # Perform attention dropout
            attention_map: torch.Tensor = self.attention_dropout(attention_map)
            # Apply attention map and reshape
            output[:, :, token_index_query] = torch.einsum("bhl, bhlv -> bhv", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, height, width]
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output tensor of the shape [batch size * windows, channels, height, width]
        """
        # Save original shape
        batch_size_windows, channels, height, width = input.shape
        tokens: int = height * width
        # Reshape input to [batch size * windows, tokens (height * width), channels]
        input: torch.Tensor = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        # Perform query, key, and value mapping
        query_key_value: torch.Tensor = self.mapping_qkv(input)
        query_key_value: torch.Tensor = query_key_value.view(batch_size_windows, tokens, 3, self.num_heads,
                                                             channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # Perform attention
        if self.sequential_self_attention:
            output: torch.Tensor = self.__sequential_self_attention(query=query, key=key, value=value,
                                                                    batch_size_windows=batch_size_windows,
                                                                    tokens=tokens,
                                                                    mask=mask)
        else:
            output: torch.Tensor = self.__self_attention(query=query, key=key, value=value,
                                                         batch_size_windows=batch_size_windows, tokens=tokens,
                                                         mask=mask)
        # Perform linear mapping and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output))
        # Reshape output to original shape [batch size * windows, channels, height, width]
        output: torch.Tensor = output.permute(0, 2, 1).view(batch_size_windows, channels, height, width)
        return output
        

class Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Num Patches, Channels
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class VajraV2Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.pyconv = DWPyConv2(dim, dim, [3, 5], True)
        #self.dwconv = DepthwiseConvBNAct(dim, dim, stride=1, kernel_size=5, groups=dim)
        #self.pwconv = ConvBNAct(dim, dim, stride=1, kernel_size=1)
        self.conv = ConvBNAct(dim, dim, stride=1, kernel_size=3)

    def forward(self, x):
        pyconv = self.pyconv(x)
        out = self.conv(pyconv)
        return x + out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_c, mid_c=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False) -> None:
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(in_c, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(in_c, mid_c)
        self.fc2 = nn.Linear(mid_c, in_c)

        self.norm1 = nn.LayerNorm(in_c)
        self.norm2 = nn.LayerNorm(in_c)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.multihead_attention(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_pdding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.multihead_attention(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_pdding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class AIFI(TransformerEncoderLayer):
    def __init__(self, in_c, mid_c=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False) -> None:
        super().__init__(in_c, mid_c, num_heads, dropout, act, normalize_before)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=1000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([torch.sin(out_w), torch.cos(out_w),
                             torch.sin(out_h), torch.cos(out_h)], axis=1)[None, :, :]

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute((0, 2, 1)).view([-1, c, h, w])

class BottleneckAttn(nn.Module):
    def __init__(self, dim, window_size, num_heads, expansion_ratio=1) -> None:
        super().__init__()
        #hidden_dim = int(out_c * expansion_ratio)
        #self.add = in_c == out_c
        self.window_size = window_size
        #self.conv1 = self.conv1 = nn.Conv2d(in_c, hidden_dim, kernel_size=3, stride=1, padding=1, groups=math.gcd(in_c, hidden_dim))
        #self.norm1 = nn.LayerNorm(hidden_dim)

        self.attention = Attention(dim, window_size=window_size, num_heads=num_heads) #AIFI(dim, mid_c=dim // 2, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(dim)
        self.act2 = nn.SiLU()

    def forward(self, x):
        input = x
        #x = self.conv1(x)
        #x = x.permute(0, 2, 3, 1)
        #x = self.norm1(x)
        #x = x.permute(0, 3, 1, 2)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # Reshape to [B, N, C], where N = H * W
        x = self.attention(x)  # Pass the reshaped tensor to attention
        x = self.norm2(x.view(B, H, W, C)).permute(0, 3, 1, 2)
        x = self.act1(x)

        x = self.conv2(x)
        #x = x.permute(0, 2, 3, 1)
        x = self.norm3(x)
        #x = x.permute(0, 3, 1, 2)
        x = self.act2(x)

        return input + x

class BottleneckWindowAttn(nn.Module):
    def __init__(self, in_c, out_c, window_size, num_heads, expansion_ratio=1., shift=True) -> None:
        super().__init__()
        hidden_dim = int(out_c * expansion_ratio)
        self.add = in_c == out_c
        self.shift = shift
        self.window_size = window_size
        self.shift_size = window_size // 2 if self.shift else 0
        self.conv1 = self.conv1 = nn.Conv2d(in_c, hidden_dim, kernel_size=3, stride=1, padding=1, groups=math.gcd(in_c, hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.attention = WindowMultiHeadAttention(hidden_dim, window_size, num_heads, meta_hidden_feats=hidden_dim if hidden_dim <= 256 else 256)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, out_c, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        self.norm3 = nn.LayerNorm(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)


        if self.shift_size > 0:
            _, _, height, width = x.shape
            mask: torch.Tensor = torch.zeros(height, width, device=self.attention.tau.device)
            height_slices: Tuple = (slice(0, -self.window_size),
                                    slice(-self.window_size, -self.shift_size),
                                    slice(-self.shift_size, None))
            width_slices: Tuple = (slice(0, -self.window_size),
                                   slice(-self.window_size, -self.shift_size),
                                   slice(-self.shift_size, None))
            counter: int = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    mask[height_slice, width_slice] = counter
                    counter += 1
            mask_windows: torch.Tensor = unfold(mask[None, None], self.window_size)
            mask_windows: torch.Tensor = mask_windows.reshape(-1, self.window_size * self.window_size)
            attention_mask: Optional[torch.Tensor] = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
            attention_mask: Optional[torch.Tensor] = attention_mask.masked_fill(attention_mask == 0, float(0.0))

            output_shift: torch.Tensor = torch.roll(input=x, shifts=(-self.shift_size, -self.shift_size),
                                                    dims=(-1, -2))
        else:
            attention_mask: Optional[torch.Tensor] = None
            output_shift: torch.Tensor = x

        B, C, H, W = x.shape
        #x = x.view(B, C, H * W).transpose(1, 2)  # Reshape to [B, N, C], where N = H * W
        x = self.attention(x, mask=attention_mask)  # Pass the reshaped tensor to attention
        #x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        if self.shift_size > 0:
            output_shift: torch.Tensor = torch.roll(input=x, shifts=(self.shift_size, self.shift_size),
                                                    dims=(-1, -2))
        else:
            output_shift: torch.Tensor = x

        x = self.norm2(output_shift.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)

        return input + x if self.add else x
        

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.) -> None:
        super().__init__()
        self.dim = dim
        self.drop_path_prob = drop_path
        self.dwconv = Conv(dim, dim, stride=1, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def get_module_info(self):
        return "ConvNeXtV2Block", f"[{self.dim}, {self.drop_path_prob}]"

class MobileNetV2InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4) -> None:
        super().__init__()
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.expand_ratio=expand_ratio
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_c * expand_ratio))
        self.add = self.stride == 1 and in_c == out_c
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_c, hidden_dim, kernel_size=1, act='relu6'))

        layers.extend(
            [
                DepthwiseConvBNAct(hidden_dim, hidden_dim, stride=stride, kernel_size=3, act="relu6"),
                nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.add else self.conv(x)

    def get_module_info(self):
        return "MobileNetInvertedResidual", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.expand_ratio}]"

class MobileNetV3InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=3, dilation=1, expand_ratio=4, use_se=False, use_hs=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = 1 if dilation > 1 else stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.use_hs = use_hs

        self.add = self.stride == 1 and in_c == out_c
        act = "hardswish" if use_hs else "relu6"

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_c * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_c, hidden_dim, kernel_size=1, stride=1, act=act))
        
        layers.append(DepthwiseConvBNAct(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation, stride=stride, act=act))

        if use_se:
            layers.append(SELayer(hidden_dim, 4))

        layers.append(ConvBNAct(hidden_dim, out_c, kernel_size=1, stride=1, act=None))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.blocks(x) if self.add else self.blocks(x)

    def get_module_info(self):
        return "MobileNetV3InvertedResidual", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.dilation}, {self.expand_ratio}, {self.use_se}, {self.use_hs}]"    

class MobileNetV4InvertedResidual(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c, 
                 expand_ratio, 
                 start_dw_kernel_size, 
                 middle_dw_kernel_size, 
                 stride,
                 middle_dw_downsample: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init_value: float = 1e-5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.expand_ratio = expand_ratio
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.stride = stride
        self.middle_dw_downsample = middle_dw_downsample
        self.layer_scale_init_value = layer_scale_init_value

        if start_dw_kernel_size:
            self.start_dw_conv = nn.Conv2d(in_c, in_c, start_dw_kernel_size, 
                                           stride if not middle_dw_downsample else 1,
                                           (start_dw_kernel_size - 1) // 2,
                                           groups=in_c, bias=False)
            self.start_dw_norm = nn.BatchNorm2d(in_c)

        expand_channels = make_divisible(in_c * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_c, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
            self.middle_dw_conv = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                            stride if middle_dw_downsample else 1,
                                            (middle_dw_kernel_size - 1) // 2,
                                            groups=expand_channels, bias=False)
            self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
            self.middle_dw_act = nn.ReLU(inplace=True)
        
        self.proj_conv = nn.Conv2d(expand_channels, out_c, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_c)

        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_c)), requires_grad=True)

        self.use_layer_scale = use_layer_scale
        self.add = stride == 1 and in_c == out_c

    def forward(self, x):
        shortcut = x
        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)
        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.add else x

    def get_module_info(self):
        return "MobileNetV4InvertedResidual", f"[{self.in_c}, {self.out_c}, {self.expand_ratio}, {self.start_dw_kernel_size}, {self.middle_dw_kernel_size}, {self.stride}, {self.middle_dw_downsample}, {self.use_layer_scale}, {self.layer_scale_init_value}]"

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"

class BNContrastiveHead(nn.Module):
    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x):
        x = self.norm(x)
        w = F.normalize(w, dims=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

class ContrastiveHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x):
        x = F.normalize(x, dims=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

class ImagePoolingAttention(nn.Module):
    def __init__(self, in_channels=(), ec=256, ct=512, num_heads=8, kernel_size=3, scale=False) -> None:
        super().__init__()
        num_features = len(in_channels)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(ch, ec, kernel_size=1) for ch in in_channels])
        self.img_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((kernel_size, kernel_size)) for _ in range(num_features)])
        self.ec = ec
        self.num_heads = num_heads
        self.num_features = num_features
        self.head_channels = ec // num_heads
        self.kernel_size = kernel_size
    
    def forward(self, x, text):
        batch_size = x[0].shape[0]
        assert len(x) == self.num_features
        num_patches = self.kernel_size ** 2
        x = [pool(proj(x)).view(batch_size, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.img_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(batch_size, -1, self.num_heads, self.head_channels)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_channels)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_channels)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.head_channels**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(batch_size, -1, self.ec))
        return x * self.scale + text

def _trunc_normal_(tensor, mean, std, a, b):
    # Taken from timm
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        import warnings
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Taken from timm
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x

class SplitAttn(nn.Module):
    def __init__(self, in_c, out_c=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.SiLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_c = out_c or in_c
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_c * radix
        if rd_channels is None:
            attn_chs = make_divisible(in_c * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_c, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_c, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        output = out.contiguous()
        return output

class BottleneckResNetSplitAttention(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, first_dilation=None, dilation=1, 
                 is_first=False,) -> None:
        super().__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        self.radix = radix
        self.conv1 = ConvBNAct(inplanes, group_width, kernel_size=1, stride=1, bias=False, act="relu")
        self.avd = avd and (stride > 1 or is_first)

        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0

        self.avd_first = avd_first

        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        if radix >= 1:
            self.conv2 = SplitAttn(
                group_width, group_width, kernel_size=3, stride=stride, 
                padding=dilation, dilation=dilation, groups=cardinality, bias=False,
                radix=radix,
            )

        else:
            self.conv2 = ConvBNAct(group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False, act="relu"
            )
        
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None
        self.conv3 = ConvBNAct(group_width, planes * 4, kernel_size=1, bias=False, act=None)
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        
        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        
        out += shortcut
        out = self.act3(out)
        return out

class MSBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_se=False, radix=2, expand_ratio=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.use_se = use_se
        self.radix = radix
        self.expand_ratio = expand_ratio

        hidden_dim = int(in_c * expand_ratio)
        self.add = stride == 1 and in_c == out_c
        if not self.identity:
            if stride > 1:
                self.down = nn.Sequential(MaxPool(3, 2, 1),
                                          Conv(in_c, out_c, 1, 1)
                                         )
            else:
                self.down = Conv(in_c, out_c, 1, 1)

        if use_se:
            self.conv = nn.Sequential(
                #pointwise
                ConvBNAct(in_c, hidden_dim, 1, 1),
                #mixconv
                MixConv2d(hidden_dim,hidden_dim, kernel_size=(3,5), stride=stride, equal_ch=True),
                #se
                SELayer(hidden_dim, reduction=self.expand_ratio*4),
                #pointwise
                ConvBNAct(hidden_dim, out_c, 1, 1, act=None)
            )

        else:
            self.conv = nn.Sequential(
                #mixconv
                MixConv2d(in_c,hidden_dim,k=(3,5,7,9),s=1,equal_ch=True),
                #split attention - radix = 2 - SK-Unit, radix = 1 - SE-Layer
                SplitAttn(hidden_dim, out_c, stride=stride,groups=1,radix=radix,
                          rd_ratio=0.25, norm_layer=nn.BatchNorm2d),
            )

    def forward(self, x):
        if self.add:
            out = x + self.conv(x)
        else:
            out = self.down(x) + self.conv(x)
        return channel_shuffle(out, 2)

    def get_module_info(self):
        return "MSBlock", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.use_se}, {self.radix}, {self.expand_ratio}]"

# Modules for Heads - 
class DistributedFocalLoss(nn.Module):
    """ https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py 
        https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, in_c=16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_c, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_c, 1, 1))
        self.in_c = in_c

    def forward(self, x):
        batch, channels, anchors = x.shape
        return self.conv(x.view(batch, 4, self.in_c, anchors).transpose(2, 1).softmax(1)).view(batch, 4, anchors)

class ProtoMaskModule(nn.Module):
    def __init__(self, in_c, c_mid = 256, out_c = 32) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_c = in_c, out_c = c_mid, kernel_size=3)
        self.conv2 = ConvBNAct(in_c = c_mid, out_c = c_mid, kernel_size=3)
        self.conv3 = ConvBNAct(in_c = c_mid, out_c=out_c)

    def forward(self, x):
        fm1 = self.conv1(x)
        _, _, H, W = fm1.shape
        H = 2 * H
        W = 2 * W
        up = F.interpolate(fm1, size=(H, W), mode="nearest")
        fm2 = self.conv2(up)
        out = self.conv3(fm2)
        return out

class UConv(nn.Module):
    def __init__(self, in_c, hidden_c = 256, out_c = 256):
        super().__init__()
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(hidden_c, out_c, 1, 1)

    def forward(self, x):
        fm1 = self.conv1(x)
        fm2 = self.conv2(fm1)
        _, _, H, W = fm2.shape
        H = 2 * H
        W = 2 * W
        out = F.interpolate(fm2, size=(H, W), mode="nearest")
        return out
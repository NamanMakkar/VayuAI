# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Tuple, Optional, List, Union, Any, Callable
from vajra.utils.torch_utils import fuse_conv_and_bn
from vajra.utils import LOGGER

try:
    import einops
except ImportError:
    einops = None

act_table = {'silu' : nn.SiLU(),
             'relu' : nn.ReLU(),
             'relu6': nn.ReLU6(),
             'hardswish': nn.Hardswish(),
             'mish': nn.Mish()}

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="relu6", p=None, g=1, d=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.act = act
        self.p = p
        self.g = g
        self.d = d
        padding = 0 if k==s else autopad(k,p,d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act_table.get(act)
    
    def forward(self, x):
        return self.act(self.bn(self.c(x)))
    
    def get_module_info(self):
        return "LeYOLO_mn_conv", f"[{self.c1}, {self.c2}, {self.k}, {self.act}, {self.p}, {self.g}, {self.d}]"

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

class RepConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=3, padding=1, groups=1, dilation=1, act="silu", bn=False, deploy=False):
        super().__init__()
        assert kernel_size == 3 and padding == 1
        self.in_c = in_c
        self.out_c = out_c
        self.groups = groups
        self.act = act_table.get(act) if act is not None else nn.Identity()
        self.bn = nn.BatchNorm2d(in_c) if bn and in_c == out_c and stride == 1 else None
        self.conv1 = ConvBNAct(in_c, out_c, stride=stride, kernel_size=kernel_size, padding=padding, groups=groups, act=None)
        self.conv2 = ConvBNAct(in_c, out_c, kernel_size=1, stride=1, padding=(padding - kernel_size // 2), groups=groups, act=None)

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBNAct):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor") 

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        square = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(square + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

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
        self.groups = groups
        self.dilation=dilation
        self.bias = bias
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
        return f"ConvBNAct", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.padding}, {self.groups}, {self.dilation}, {self.bias}, {self.act}]"

class ConvDyTAct(nn.Module):
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
        self.groups = groups
        self.dilation=dilation
        self.bias = bias
        self.conv = nn.Conv2d(in_c, 
                              out_c, 
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.dyt = DynamicTanh(out_c, False)
        self.act = act_table.get(act) if act is not None else nn.Identity()
    
    def forward(self, x):
        return self.act(self.dyt(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def get_module_info(self):
        return f"ConvBNAct", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.padding}, {self.groups}, {self.dilation}, {self.bias}, {self.act}]"

class ConvMakkarNorm(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=1, padding=None, groups=1, dilation=1, bias=False) -> None:
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
        self.groups = groups
        self.dilation=dilation
        self.bias = bias
        self.conv = nn.Conv2d(in_c,
                              out_c,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.mn = MakkarNormalization(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.mn(self.conv(x)))
    
    def get_module_info(self):
        return "ConvMakkarNorm", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.kernel_size}, {self.padding}, {self.groups}, {self.dilation}, {self.bias}]"

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
    def extra_repr(self):
        return "{num_features}, eps={eps}".format(**self.__dict__)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor):
        output = torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
        return output * self.weight.view(1, -1, 1, 1)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

class DynamicTanh(nn.Module):
    def __init__(self, channels, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class MakkarNormalization(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.k = nn.Parameter(torch.ones(1) * 2.0)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = 2 / (1 + torch.exp(-self.k * x)) - 1
        return x * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)

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
    """ Pyramidal Convolutional Block with 2 DW convolutions """
    def __init__(self, in_c, out_c,  pyconv_kernels=[3, 5], shortcut=False):
        super(DWPyConv2, self).__init__()
        assert in_c == out_c, "In channels and out channels must be equal for DWPyConv2"
        self.conv1_1 = nn.Sequential(DepthwiseConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[0]), ConvBNAct(out_c // 2, out_c // 2, 1, 1))
        self.conv1_2 = nn.Sequential(DepthwiseConvBNAct(in_c, out_c//2, kernel_size=pyconv_kernels[1]), ConvBNAct(out_c // 2, out_c // 2, 1, 1))
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        pyramid = torch.cat((self.conv1_1(x), self.conv1_2(x)), dim=1)
        return x + pyramid if self.add else pyramid
    
class RepVGGDWPyConv2(nn.Module):
    """ Pyramidal Convolutional Block with 2 DW convolutions """
    def __init__(self, in_c, out_c, shortcut=False, rep_vgg_k=5):
        super(RepVGGDWPyConv2, self).__init__()
        assert in_c == out_c, "In channels and out channels must be equal for DWPyConv2"
        self.conv1_1 = nn.Sequential(DepthwiseConvBNAct(in_c, in_c, kernel_size=3), ConvBNAct(in_c, out_c // 2, 1, 1))
        self.conv1_2 = nn.Sequential(RepVGGDW(in_c, kernel_size=rep_vgg_k), ConvBNAct(in_c, out_c // 2, 1, 1))
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        pyramid = torch.cat((self.conv1_1(x), self.conv1_2(x)), dim=1)
        return x + pyramid if self.add else pyramid

class DWPyConv3(nn.Module):
    """ Pyramidal Convolutional Block with 3 DW convolutions"""
    def __init__(self, in_c, out_c,  pyconv_kernels=[3, 5, 7], shortcut=False):
        super(DWPyConv3, self).__init__()
        assert in_c == out_c, "In channels and out channels must be equal for DWPyConv3"
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
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.expansion_ratio=expansion_ratio
        self.groups=groups
        self.act = act
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))
    
    def get_module_info(self):
        return "Bottleneck", f"[{self.in_c}, {self.out_c}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.groups}, {act_table[self.act]}]"

class RepBottleneck(nn.Module):
    """Standard bottleneck"""

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3,3), expansion_ratio=0.5, groups=1, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.expansion_ratio=expansion_ratio
        self.groups=groups
        self.act = act
        self.conv1 = RepConv(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act) #ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))
    
    def get_module_info(self):
        return "RepBottleneck", f"[{self.in_c}, {self.out_c}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.groups}, {act_table[self.act]}]"

class BottleneckMakkarNorm(nn.Module):
    """Standard bottleneck"""

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3,3), expansion_ratio=0.5, groups=1):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.expansion_ratio=expansion_ratio
        self.groups = groups
        self.conv1 = ConvMakkarNorm(in_c, hidden_c, kernel_size=kernel_size[0], stride=1)
        self.conv2 = ConvMakkarNorm(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))
    
    def get_module_info(self):
        return "BottleneckMakkarNorm", f"[{self.in_c}, {self.out_c}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.groups}]"

class BottleneckDW(nn.Module):
    """Bottleneck with 2x 3x3 convolutions and 2x 3x3 DW convolutions"""

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3,3), expansion_ratio=0.5, groups=1, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.expansion_ratio=expansion_ratio
        self.groups=groups
        self.act = act
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act)
        self.conv2 = DepthwiseConvBNAct(hidden_c, hidden_c, 1, 3)
        self.conv3 = DepthwiseConvBNAct(hidden_c, hidden_c, 1, 3)
        self.conv4 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.conv3(self.conv2(self.conv1(x)))) if self.add else self.conv2(self.conv3(self.conv2(self.conv1(x))))
    
    def get_module_info(self):
        return "BottleneckDW", f"[{self.in_c}, {self.out_c}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.groups}, {act_table[self.act]}]"

class RepBottleneckDW(nn.Module):
    """Bottleneck with 2x 3x3 convolutions and 2x RepVGGDW convolutions"""

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=(3,3), expansion_ratio=0.5, groups=1, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.expansion_ratio=expansion_ratio
        self.groups=groups
        self.act = act
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act)
        self.conv2 = RepVGGDW(hidden_c)
        self.conv3 = RepVGGDW(hidden_c)
        self.conv4 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.conv2(self.conv3(self.conv2(self.conv1(x)))) if self.add else self.conv2(self.conv3(self.conv2(self.conv1(x))))
    
    def get_module_info(self):
        return "RepBottleneckDW", f"[{self.in_c}, {self.out_c}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.groups}, {act_table[self.act]}]"

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
    """Standard bottleneck"""

    def __init__(self, in_c, out_c, kernel_size=(3,3), expansion_ratio=0.5, groups=1, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)  # hidden channels
        assert in_c == out_c, "in channels and out channels must be equal for residual bottleneck"
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size=kernel_size
        self.expansion_ratio=expansion_ratio
        self.groups=groups
        self.act = act
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=kernel_size[0], stride=1, act=act)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=kernel_size[1], stride=1, groups=groups, act=act)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))
    
    def get_module_info(self):
        return "BottleneckV2", f"[{self.in_c}, {self.out_c}, {self.kernel_size}, {self.expansion_ratio}, {self.groups}, {act_table[self.act]}]"

class MSBottleneck(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        hidden_c = int(out_c * 0.5)
        assert in_c == out_c, "in channels and out channels must be equal for MSBottleneck"
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = DepthwiseConvBNAct(hidden_c, hidden_c, 1, kernel_size=kernel_size)
        self.conv3 = ConvBNAct(hidden_c, out_c, 1, 1)

    def forward(self, x):
        return x + self.conv3(self.conv2(self.conv1(x)))
    
    def get_module_info(self):
        return "MSBottleneck", f"[{self.in_c}, {self.out_c}, {self.kernel_size}]"

class BottleneckV3(nn.Module):

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=3, expansion_ratio=0.5, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=3, stride=1, act=act)
        self.dw_conv = ConvBNAct(hidden_c, hidden_c, kernel_size=kernel_size, stride=1, groups=hidden_c, act=act)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=3, stride=1, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        conv1 = self.conv1(x)
        dw_conv1 = self.dw_conv(conv1)
        conv2 = self.conv2(dw_conv1)
        return x + conv2 if self.add else conv2
    
class BottleneckV4(nn.Module):

    def __init__(self, in_c, out_c, shortcut=True, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.conv1 = RepVGGDW_3x3(in_c, 3, 1)
        self.conv2 = ConvBNAct(in_c, out_c, kernel_size=3, stride=1, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return x + conv2 if self.add else conv2

class BottleneckSplitAttn(nn.Module):

    def __init__(self, in_c, out_c, shortcut=True, kernel_size=3, expansion_ratio=0.5, act="silu"):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=3, stride=1, act=act)
        self.split_attn = SplitAttn(hidden_c, hidden_c, 3, 1, radix=2)
        self.conv2 = ConvBNAct(hidden_c, out_c, kernel_size=3, stride=1, act=act)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        conv1 = self.conv1(x)
        splatt = self.split_attn(conv1)
        conv2 = self.conv2(splatt)
        return x + conv2 if self.add else conv2

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

class RepVGGDW(nn.Module):
    def __init__(self, dim, kernel_size=7, stride=1) -> None:
        super().__init__()
        self.kernel_size=kernel_size
        self.stride = stride
        self.conv = DepthwiseConvBNAct(dim, dim, stride=stride, kernel_size=kernel_size, act=None)
        self.conv1 = DepthwiseConvBNAct(dim, dim, stride=stride, kernel_size=3, act=None)
        self.dim = dim
        self.act = nn.SiLU()
        self.fused_conv = None  # Placeholder for fused layer

    def forward(self, x):
        if self.fused_conv is not None:
            return self.act(self.fused_conv(x))
        return self.act(self.conv(x) + self.conv1(x))
    
    def _calculate_padding(self, kernel, target_size):
        """Calculate padding required to resize the kernel to the target size."""
        current_size = kernel.shape[2]
        pad_total = target_size - current_size
        pad = pad_total // 2
        return [pad, pad + (pad_total % 2), pad, pad + (pad_total % 2)]
    
    @torch.no_grad()
    def fuse(self):
        # Fuse conv and batchnorm layers for self.conv and self.conv1
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        # Dynamically adjust padding for any kernel size
        if conv_w.shape[2:] != conv1_w.shape[2:]:
            larger_size = max(conv_w.shape[2], conv1_w.shape[2])
            conv_w = torch.nn.functional.pad(conv_w, self._calculate_padding(conv_w, larger_size))
            conv1_w = torch.nn.functional.pad(conv1_w, self._calculate_padding(conv1_w, larger_size))

        # Combine the two weights and biases
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        # Create a new fused_conv layer with combined weights and biases
        device = next(self.parameters()).device
        self.fused_conv = nn.Conv2d(
            in_channels=self.conv.conv.in_channels,
            out_channels=self.conv.conv.out_channels,
            kernel_size=self.conv.conv.kernel_size,
            stride=self.conv.conv.stride,
            padding=self.conv.conv.padding,
            groups=self.conv.conv.groups,
            bias=True
        ).to(device)
        
        self.fused_conv.weight.data.copy_(final_conv_w.to(device))
        self.fused_conv.bias.data.copy_(final_conv_b.to(device))

    def forward_fuse(self, x):
        return self.act(self.fused_conv(x)) if self.fused_conv is not None else self.act(self.conv(x) + self.conv1(x))

    def get_module_info(self):
        return "RepVGGDW", f"[{self.dim}, {self.kernel_size}, {self.stride}]"

class MerudandaDW(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, expansion_ratio=0.5, use_rep_vgg_dw=False, kernel_size=5, stride=1):
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.expansion_ratio = expansion_ratio
        self.use_rep_vgg_dw = use_rep_vgg_dw
        self.kernel_size = kernel_size
        self.stride = stride
        self.add = shortcut and in_c == out_c and stride == 1
        self.block = nn.Sequential(
            DepthwiseConvBNAct(in_c, in_c, kernel_size=3),
            ConvBNAct(in_c, 2 * hidden_c, 1, 1),
            DepthwiseConvBNAct(2 * hidden_c, 2 * hidden_c, stride=stride, kernel_size=kernel_size) if not use_rep_vgg_dw else RepVGGDW(2 * hidden_c, kernel_size=kernel_size, stride=stride),
            ConvBNAct(2 * hidden_c, out_c, 1, 1),
            DepthwiseConvBNAct(out_c, out_c, 1, 3),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.add else y
    
    def get_module_info(self):
        return "MerudandaDW", f"[{self.in_c}, {self.out_c}, {self.shortcut}, {self.expansion_ratio}, {self.use_rep_vgg_dw}, {self.kernel_size}, {self.stride}]"
    
class VajraMerudandaMS(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(1, 3, 3), num_blocks=2, expansion_ratio=0.5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size=kernel_size
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int((out_c * expansion_ratio) * len(kernel_size))
        self.block_c = self.hidden_c // len(kernel_size)
        self.conv1 = ConvBNAct(in_c, self.hidden_c, 1, 1)
        self.blocks = []
        for i in range(len(kernel_size)):
            if i == 0:
                self.blocks.append(nn.Identity())
                continue
            block = nn.Sequential(*[MerudandaDW(self.block_c, self.block_c, True, 0.5, use_rep_vgg_dw=True, kernel_size=kernel_size[i]) for _ in range(num_blocks)])
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
        self.conv2 = ConvBNAct(self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        fms = []

        for i, block in enumerate(self.blocks):
            fm = conv1[:,i*self.block_c:(i+1)*self.block_c,...]
            if i > 1:
                fm = fm + fms[i-1]
            fm = block(fm)
            fms.append(fm)
        block_out = torch.cat(fms, 1)
        out = self.conv2(block_out)
        return out
    
    def get_module_info(self):
        return "VajraMerudandaMS", f"[{self.in_c}, {self.out_c}, {self.kernel_size}, {self.num_blocks}, {self.expansion_ratio}]"

class VajraGrivaV2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, inner_block=False, use_cbam=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int((out_c * expansion_ratio) * 2)
        self.use_cbam = use_cbam
        self.block_c = self.hidden_c // 2
        self.conv1 = ConvBNAct(in_c // 2, self.hidden_c, 1, 1)
        block = InnerBlock if inner_block else Bottleneck
        self.blocks = nn.ModuleList(
                block(self.block_c, self.block_c, 2, True, 1, 0.5) for _ in range(num_blocks)
            ) if block == InnerBlock else nn.ModuleList(
                block(self.block_c, self.block_c, shortcut=True, expansion_ratio=0.5) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct(in_c + (num_blocks + 2) * self.block_c, out_c, 1, 1)
        self.cbam = CBAM(out_c) if use_cbam else nn.Identity()
        self.add = in_c == out_c

    def forward(self, x):
        in_1, in_2 = x.chunk(2, 1)
        in_2 = in_2 + in_1
        conv1 = self.conv1(in_2)
        fm1, fm2 = conv1.chunk(2, 1)
        #fm1 = conv1[:, 0:self.block_c, ...]
        #fm2 = conv1[:, self.block_c:2*self.block_c, ...]
        fm2 = fm2 + fm1
        fms = [in_1, in_2, fm1, fm2]
        fms.extend(block(fms[-1]) for block in self.blocks)
        block_out = torch.cat(fms, 1)
        out = self.conv2(block_out)
        cbam = self.cbam(out)
        return cbam + x if self.add else cbam + out
    
    def get_module_info(self):
        return "VajraGrivaV2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.use_cbam}]"

class VajraMerudandaV2Bhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, inner_block=False, kernel_size=3, use_cbam=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.block_c = self.in_c // 2
        self.inner_block = inner_block
        self.kernel_size = kernel_size
        self.use_cbam = use_cbam
        block = InnerBlock if inner_block else Bottleneck
        self.conv1 = ConvBNAct(in_c, in_c, 1, 1)
        self.blocks = nn.ModuleList(
                block(self.block_c, self.block_c, 2, True, 1, 0.5) for _ in range(num_blocks)
            ) if block == InnerBlock else nn.ModuleList(
                block(self.block_c, self.block_c, shortcut=True, expansion_ratio=0.5) for _ in range(num_blocks)
            )
        self.dwconv = nn.Sequential(DepthwiseConvBNAct(self.block_c, self.block_c, 1, kernel_size=kernel_size), ConvBNAct(self.block_c, self.block_c, 1, 1))
        self.conv2 = ConvBNAct(in_c + num_blocks * self.block_c, out_c, 1, 1)
        self.add = in_c == out_c
        self.cbam = CBAM(out_c) if use_cbam else nn.Identity()

    def forward(self, x):
        conv1 = self.conv1(x)
        fm1, fm2 = conv1.chunk(2, 1)
        fm1 = self.dwconv(fm1)
        fm2 = fm2 + fm1
        fms = [fm1, fm2]
        fms.extend(block(fms[-1]) for block in self.blocks)
        out = self.conv2(torch.cat(fms, 1))
        cbam = self.cbam(out)
        return x + cbam if self.add else cbam + out
    
    def get_module_info(self):
        return "VajraMerudandaV2Bhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.inner_block}, {self.kernel_size}, {self.use_cbam}]"

class VajraGrivaV2Bhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, inner_block=False, use_cbam=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.inner_block = inner_block
        block = InnerBlock if inner_block else Bottleneck
        self.block_c = in_c // 4
        self.use_cbam = use_cbam
        self.blocks = nn.ModuleList(
                block(self.block_c, self.block_c, 2, True, 1, 0.5) for _ in range(num_blocks)
            ) if block == InnerBlock else nn.ModuleList(
                block(self.block_c, self.block_c, shortcut=True, expansion_ratio=0.5) for _ in range(num_blocks)
            )
        self.conv = ConvBNAct(in_c + num_blocks * self.block_c, out_c, 1, 1)
        self.dwconv1 = nn.Sequential(ConvBNAct(in_c // 4, in_c // 4, 1, 1), DepthwiseConvBNAct(in_c // 4, in_c // 4, 1, 3), ConvBNAct(in_c // 4, in_c // 4, 1, 1))
        self.dwconv2 = nn.Sequential(ConvBNAct(in_c // 4, in_c // 4, 1, 1), DepthwiseConvBNAct(in_c // 4, in_c // 4, 1, 3), ConvBNAct(in_c // 4, in_c // 4, 1, 1))
        self.cbam = CBAM(out_c) if use_cbam else nn.Identity()

    def forward(self, inputs):
        fm1, fm2, fm3, fm4 = inputs.chunk(4, 1)
        fm2 = fm2 + fm1
        fm2 = self.dwconv1(fm2)
        fm3 = fm3 + fm2
        fm3 = self.dwconv2(fm3)
        fm4 = fm4 + fm3
        fms = [fm1, fm2, fm3, fm4]
        fms.extend(block(fms[-1]) for block in self.blocks)
        out = self.conv(torch.cat(fms, 1))
        cbam = self.cbam(out)
        return cbam + out
    
    def get_module_info(self):
        return "VajraGrivaV2Bhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.inner_block}, {self.use_cbam}]"

class VajraGrivaV2Bhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, inner_block=False, use_cbam=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.inner_block = inner_block
        block = InnerBlock if inner_block else Bottleneck
        self.block_c = out_c // 2
        self.hidden_c = in_c - 3 * self.block_c
        self.use_cbam = use_cbam
        self.blocks = nn.ModuleList(
                block(self.block_c, self.block_c, 2, True, 1, 0.5) for _ in range(num_blocks)
            ) if block == InnerBlock else nn.ModuleList(
                block(self.block_c, self.block_c, shortcut=True, expansion_ratio=0.5) for _ in range(num_blocks)
            )
        self.dwconv = nn.ModuleList(nn.Sequential(ConvBNAct(self.block_c, self.block_c, 1, 1), DepthwiseConvBNAct(self.block_c, self.block_c, 1, 3), ConvBNAct(self.block_c, self.block_c, 1, 1)) for _ in range(2))
        self.conv = ConvBNAct(in_c + num_blocks * self.block_c, out_c, 1, 1)
        self.cbam = CBAM(out_c) if use_cbam else nn.Identity()

    def forward(self, inputs):
        fms = list(inputs.split(self.block_c, dim=1))
        for i in range(len(fms)):                
            if i >= 1:
                fms[i] = fms[i] + fms[i-1]
                #if i < len(fms) - 1:
                    #fms[i] = self.dwconvs[i-1](fms[i])
            if i == len(fms) - 3:
                fms[i] = self.dwconv[0](fms[i])

            if i == len(fms) - 2:
                fms[i] = self.dwconv[1](fms[i])

        fms.extend(block(fms[-1]) for block in self.blocks)
        out = self.conv(torch.cat(fms, 1))
        cbam = self.cbam(out)
        return cbam + out

    def get_module_info(self):
        return "VajraGrivaV2Bhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.inner_block}, {self.use_cbam}]"

class AttentionBottleneckV5(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden_c = int((out_c * expansion_ratio) * 2)
        self.block_c = self.hidden_c // 2
        self.conv1 = ConvBNAct(in_c, self.hidden_c, 1, 1)
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.attn = nn.ModuleList(
                [nn.Identity()] + 
                [nn.Sequential(*[AttentionBlock(self.block_c, self.block_c, num_heads=self.block_c // 64) for _ in range(num_blocks)])]
            )
        self.conv2 = ConvBNAct(self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        fms = []

        for i, block in enumerate(self.attn):
            fm = conv1[:,i*self.block_c:(i+1)*self.block_c,...]
            if i > 1:
                fm = fm + fms[i-1]
            fm = block(fm)
            fms.append(fm)
        block_out = torch.cat(fms, 1)
        out = self.conv2(block_out)
        return out
    
    def get_module_info(self):
        return f"AttentionBottleneckV5", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}]"
    
class AttentionBottleneckV6(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        assert in_c == out_c, "For AttentionBottleneckV6 in channels should be equal to out channels"
        self.block_c = out_c // 2
        self.attn = nn.ModuleList(AttentionBlock(self.block_c, self.block_c, num_heads=self.block_c // 64) for _ in range(num_blocks))
        self.dwconv = nn.Sequential(ConvBNAct(self.block_c, self.block_c, 1, 1), DepthwiseConvBNAct(self.block_c, self.block_c, 1, 3), ConvBNAct(self.block_c, self.block_c, 1, 1))
        self.conv = ConvBNAct(in_c + num_blocks * self.block_c, out_c, 1, 1)

    def forward(self, x):
        fm1, fm2 = x.chunk(2, 1)
        fm1 = self.dwconv(fm1)
        fm2 = fm2 + fm1
        fms = [fm1, fm2]
        fms.extend(attn_block(fms[-1]) for attn_block in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out + x
    
    def get_module_info(self):
        return "AttentionBottleneckV6", f"[{self.in_c}, {self.out_c}, {self.num_blocks}]"

class ADown(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden_c = out_c // 2
        self.conv1 = ConvBNAct(in_c // 2, self.hidden_c, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNAct(in_c // 2, self.hidden_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.conv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), 1)
    
    def get_module_info(self):
        return "ADown", f"[{self.in_c}, {self.out_c}]"
    
class SCDown(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.pw = ConvBNAct(in_c, out_c, 1, 1)
        self.dw = ConvBNAct(out_c, out_c, stride=stride, kernel_size=kernel_size, groups=out_c, act=None)

    def forward(self, x):
        return self.dw(self.pw(x))
    
    def get_module_info(self):
        return "SCDown", f"[{self.in_c}, {self.out_c}, {self.kernel_size}, {self.stride}]"
    
class SCDown2(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.pw = ConvBNAct(in_c, out_c, 1, 1)
        self.dw = ConvBNAct(out_c, out_c, stride=stride, kernel_size=kernel_size, act=None)

    def forward(self, x):
        return self.dw(self.pw(x))
    
    def get_module_info(self):
        return "SCDown", f"[{self.in_c}, {self.out_c}, {self.kernel_size}, {self.stride}]"
    
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

class VajraMerudandaBhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, bottleneck_dwcib=False, expansion_ratio=0.5, dw=False, use_cbam=False, use_rep_vgg_dw=False) -> None:
        super().__init__()
        block = MerudandaDW if bottleneck_dwcib else Bottleneck
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.bottleneck_dwcib = bottleneck_dwcib
        self.dwconv = dw
        self.use_cbam = use_cbam
        self.use_rep_vgg_dw = use_rep_vgg_dw
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size) if not dw else nn.Sequential(DepthwiseConvBNAct(in_c, in_c, 1, 3), ConvBNAct(in_c, hidden_c, 1, 1))
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0, use_rep_vgg_dw=use_rep_vgg_dw) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return x + y if self.add else y

    def get_module_info(self):
        return f"VajraMerudandaBhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_dwcib}, {self.expansion_ratio}, {self.dwconv}, {self.use_cbam}, {self.use_rep_vgg_dw}]"

class VajraMerudandaBhag3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_cbam = False, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.use_cbam = use_cbam
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        block = InnerBlock if inner_block else Bottleneck
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, 2, True, 1, 0.5) for _ in range(num_blocks)) if block == InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) #nn.ModuleList(block(hidden_c, hidden_c, 2, True, kernel_size=kernel_size, expansion_ratio=0.5, bhag1=bhag1) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return x + cbam if self.add else y + cbam

    def get_module_info(self):
        return f"VajraMerudandaBhag3", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_cbam}, {self.inner_block}]"

class VajraV1Merudanda(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_cbam = False, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.use_cbam = use_cbam
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        block = InnerBlock if inner_block else Bottleneck
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, 2, True, 1, 0.5) for _ in range(num_blocks)) if block == InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) #nn.ModuleList(block(hidden_c, hidden_c, 2, True, kernel_size=kernel_size, expansion_ratio=0.5, bhag1=bhag1) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return x + y if self.add else y

    def get_module_info(self):
        return f"VajraV1Merudanda", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_cbam}, {self.inner_block}]"

class VajraV1MerudandaBhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_cbam = False, inner_block=False, bottleneck_dw=False, scaling=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.use_cbam = use_cbam
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        block = InnerBlock if inner_block else Bottleneck
        self.scaling = scaling
        self.add = in_c == out_c
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, 2, True, 1, 0.5, bottleneck_dwcib=bottleneck_dw) for _ in range(num_blocks)) if block == InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_cbam}, {self.inner_block}]"
    
class VajraV1MerudandaBhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        self.bottleneck_blocks = nn.Sequential(*(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        #else:
            #self.bottleneck_blocks = nn.Sequential(*(AreaAttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 32, kernel_size=7, shortcut=True, area=grid_size) for _ in range(2 * num_blocks)))
            
        self.conv2 = ConvBNAct(2 * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        return self.conv2(torch.cat((a, b), 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"
    
class VajraV1MerudandaBhag4(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, rep_bottleneck=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        if inner_block:
            block = VajraV2InnerBlock
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size, rep_bottleneck=rep_bottleneck) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MerudandaBhag7(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, num_bottleneck_blocks=2, bottleneck_expansion_ratio=1.0, rep_bottleneck=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        if inner_block:
            block = VajraV2InnerBlock
        
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck

        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size, num_bottleneck_blocks=num_bottleneck_blocks, bottleneck_expansion=bottleneck_expansion_ratio, rep_bottleneck=rep_bottleneck) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [x, a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [x, a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag7", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MerudandaBhag8(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, num_bottleneck_blocks=2, bottleneck_expansion_ratio=1.0, rep_bottleneck=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        assert in_c == 2 * hidden_c, "in_c must be equal to 2 * hidden_c for fusion"
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        #block = VajraV2InnerBlock if inner_block else RepBottleneck
        if inner_block:
            block = VajraV2InnerBlock
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size, num_bottleneck_blocks=num_bottleneck_blocks, bottleneck_expansion=bottleneck_expansion_ratio, rep_bottleneck=rep_bottleneck) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag8", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MakkarNormMerudandaBhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvMakkarNorm(in_c, hidden_c, 1, kernel_size)
        block = InnerBlockMakkarNorm if inner_block else BottleneckMakkarNorm
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, 2, True, 1) for _ in range(num_blocks))) if block == InnerBlockMakkarNorm else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvMakkarNorm((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MakkarNormMerudandaBhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MakkarNormMerudandaBhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvMakkarNorm(in_c, 2 * hidden_c, 1, kernel_size)
        block = InnerBlockMakkarNorm if inner_block else BottleneckMakkarNorm
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, 2, True) for _ in range(num_blocks))) if block == InnerBlockMakkarNorm else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvMakkarNorm((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        b = b + a
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        b = b + a
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MakkarNormMerudandaBhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}]"

class VajraV1MerudandaBhag3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 3 * hidden_c, 1, kernel_size)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        self.bottleneck_blocks_branch1 = nn.Sequential(*(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks // 2))) if block == VajraV2InnerBlock else nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks // 2)))
        self.bottleneck_blocks_branch2 = nn.Sequential(*(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks // 2))) if block == VajraV2InnerBlock else nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks // 2)))
        self.conv2 = ConvBNAct(3 * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        branch1, branch2, branch3 = self.conv1(x).chunk(3, 1)
        bottleneck_branch2 = self.bottleneck_blocks_branch1(branch2)
        branch3 = branch3 + bottleneck_branch2
        bottleneck_branch3 = self.bottleneck_blocks_branch2(branch3)
        return self.conv2(torch.cat((branch1, bottleneck_branch2, bottleneck_branch3), 1))
    
    def forward_split(self, x):
        branch1, branch2, branch3 = self.conv1(x).split(self.hidden_c, 1)
        bottleneck_branch2 = self.bottleneck_blocks_branch1(branch2)
        branch3 = branch3 + bottleneck_branch2
        bottleneck_branch3 = self.bottleneck_blocks_branch2(branch3)
        return self.conv2(torch.cat((branch1, bottleneck_branch2, bottleneck_branch3), 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag3", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MerudandaBhag5(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio = 0.5, mlp_ratio=1.5, area=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.mlp_ratio = mlp_ratio
        self.area = area
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 3 * hidden_c, 1, 1)
        self.bottleneck_blocks_branch1 = nn.Sequential(InnerBlock(self.hidden_c, self.hidden_c, 2, True, 1, 0.5, False), AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=hidden_c // 64, area=area, mlp_ratio=mlp_ratio)) #nn.Sequential(*(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks // 2))) if block == VajraV2InnerBlock else nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks // 2)))
        self.bottleneck_blocks_branch2 = nn.Sequential(InnerBlock(self.hidden_c, self.hidden_c, 2, True, 1, 0.5, False), AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=hidden_c // 64, area=area, mlp_ratio=mlp_ratio)) #nn.Sequential(*(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks // 2))) if block == VajraV2InnerBlock else nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks // 2)))
        self.conv2 = ConvBNAct(5 * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        branch1, branch2, branch3 = self.conv1(x).chunk(3, 1)
        bottleneck_branch1 = self.bottleneck_blocks_branch1(branch2)
        branch3 = branch3 + bottleneck_branch1
        bottleneck_branch2 = self.bottleneck_blocks_branch2(branch3)
        return self.conv2(torch.cat((branch1, branch2, branch3, bottleneck_branch1, bottleneck_branch2), 1))
    
    def forward_split(self, x):
        branch1, branch2, branch3 = self.conv1(x).split(self.hidden_c, 1)
        bottleneck_branch1 = self.bottleneck_blocks_branch1(branch2)
        branch3 = branch3 + bottleneck_branch1
        bottleneck_branch2 = self.bottleneck_blocks_branch2(branch3)
        return self.conv2(torch.cat((branch1, branch2, branch3, bottleneck_branch1, bottleneck_branch2), 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag5", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.mlp_ratio}, {self.area}]"

class VajraV1MerudandaBhag6(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, num_bottleneck_blocks=2, bottleneck_expansion_ratio=1.0, rep_bottleneck=True, main_bottleneck_exp=0.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        #block = VajraV2InnerBlock if inner_block else RepBottleneck
        if inner_block:
            block = VajraV2InnerBlock
        
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck

        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size, num_bottleneck_blocks=num_bottleneck_blocks, bottleneck_expansion=bottleneck_expansion_ratio, rep_bottleneck=rep_bottleneck) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=main_bottleneck_exp) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag6", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV2MerudandaBhag4(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_attn = False, use_bottleneck_v3 = False, inner_block=False, grid_size=1, scaling=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_attn = use_attn
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        init_value = 0.01
        self.scaling = scaling
        self.gamma = init_value * torch.ones((out_c), requires_grad=True) if scaling else None
        if not use_attn:
            self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, use_attn=False, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks)) if block == VajraV2InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks))
        else:
            self.bottleneck_blocks = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 32, kernel_size=7, shortcut=True, area=grid_size) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return x + self.gamma.view(1, -1, 1, 1) * y if self.gamma is not None else y

    def get_module_info(self):
        return f"VajraV2MerudandaBhag4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_attn}, {self.use_bottleneck_v3}, {self.inner_block}]"

class VajraV2MerudandaBhag5(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_attn = False, use_bottleneck_v3 = False, inner_block=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_attn = use_attn
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks)) if block == VajraV2InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks))
        self.conv3 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        #y = [self.conv1(x)]
        #y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        #y = self.conv2(torch.cat(y, 1))
        a = self.conv1(x)
        b = self.conv2(x)
        y = [a]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv3(torch.cat((*y[1:], b), dim=1))
        return y

    def get_module_info(self):
        return f"VajraV2MerudandaBhag5", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_attn}, {self.use_bottleneck_v3}, {self.inner_block}]"
    
class VajraV2MerudandaBhag6(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_attn = False, use_bottleneck_v3 = False, inner_block=False, grid_size=1, scaling=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_attn = use_attn
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        init_value = 0.01
        self.scaling = scaling
        self.gamma = init_value * torch.ones((out_c), requires_grad=True) if scaling else None
        if not use_attn:
            self.bottleneck_blocks = nn.Sequential(*(block(hidden_c, hidden_c, use_attn=False, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        else:
            self.bottleneck_blocks = nn.Sequential(*(AreaAttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 32, kernel_size=7, shortcut=True, area=grid_size) for _ in range(2 * num_blocks)))
            
        self.conv2 = ConvBNAct(2 * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        #y = [self.conv1(x)]
        #y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        y = self.conv2(torch.cat((a, b), 1))
        return x + self.gamma.view(1, -1, 1, 1) * y if self.gamma is not None else y

    def get_module_info(self):
        return f"VajraV2MerudandaBhag4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_attn}, {self.use_bottleneck_v3}, {self.inner_block}]"

class VajraV2MerudandaBhag7(nn.Module):
    def __init__(self, in_c, out_c, stride=1, num_blocks=2, num_blocks_attn=1, inner_block=False, lite=False, area=1, use_attn=True, use_sppf=False, expansion_ratio=0.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.hidden_c = int(out_c * expansion_ratio)
        self.branch_a_channels = in_c - self.out_c
        self.out_c = out_c
        self.lite = lite
        self.use_attn = use_attn
        block = VajraV2InnerBlock if inner_block else Bottleneck
        self.use_sppf = use_sppf
        self.conv1 = ConvBNAct(self.out_c, 2 * self.hidden_c, 1, 1) if not self.use_sppf else SPPF(2 * self.hidden_c, 2 * self.hidden_c, 5)
        self.blocks = nn.ModuleList((block(self.hidden_c, self.hidden_c, use_attn=False, use_bottleneck_v3=False, shortcut=True, grid_size=1) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(self.hidden_c, self.hidden_c, shortcut=True, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.attn = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=2.0, area=area) for _ in range(2))) for _ in range(num_blocks_attn)
            ) if use_attn else nn.Identity()
        self.conv2 = ConvBNAct(self.branch_a_channels + (num_blocks + num_blocks_attn + 2) * self.hidden_c, out_c, 1, 1) if self.use_attn else ConvBNAct(self.branch_a_channels + (num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        conv1 = self.conv1(fm2)
        fm3, fm4 = conv1.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(block(fms[-1]) for block in self.blocks)
        if self.use_attn:
            fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv2(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"VajraV2MerudandaBhag7", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"
    
class VajraV2MerudandaBhag8(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, num_blocks_attn=1, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.num_blocks_attn = num_blocks_attn
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        self.use_attn = use_attn
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=False, use_bottleneck_v3=False, shortcut=True) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.attn = nn.ModuleList(
                (AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64, kernel_size=7, mlp_ratio=2.0, area=grid_size)) for _ in range(num_blocks_attn)
            ) if use_attn else nn.Identity()
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, 2 * hidden_c, kernel_size=1, stride=1)
        self.conv3 = ConvBNAct((num_blocks_attn + 2) * hidden_c, out_c, kernel_size=1, stride=1) if use_attn else nn.Identity()
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        b = b + a
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        if self.use_attn:
            fm1, fm2 = conv2.chunk(2, 1)
            fms_attn = [fm1, fm2]
            fms_attn.extend(attn(fms_attn[-1]) for attn in self.attn)
            return self.conv3(torch.cat(fms_attn, 1))
        return conv2
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        b = b + a
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        if self.use_attn:
            fm1, fm2 = conv2.split(self.hidden_c, 1)
            fms_attn = [fm1, fm2]
            fms_attn.extend(attn(fms_attn[-1]) for attn in self.attn)
            return self.conv3(torch.cat(fms_attn, 1))
        return conv2

    def get_module_info(self):
        return f"VajraV2MerudandaBhag8", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.num_blocks_attn}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"
    
class VajraV2MerudandaBhag9(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, grid_size=1, use_sppf = False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size) if not use_sppf else SPPF(in_c, 2 * self.hidden_c, 5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64, kernel_size=7, mlp_ratio=2.0, area=grid_size) for _ in range(2))) for _ in range(num_blocks)
            ) 
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        b = b + a
        y = [a, b]
        y.extend(attn(y[-1]) for attn in self.attn)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        b = b + a
        y = [a, b]
        y.extend(attn(y[-1]) for attn in self.attn)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV2MerudandaBhag9", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}]"

class VajraV2MerudandaBhag10(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=True, expansion_ratio=0.5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, 1)
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, 1, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(*(InnerBlock(hidden_c, hidden_c, 2, True, 1, 0.5) for _ in range(2))) for _ in range(num_blocks)
        )
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = self.conv1(x).split((self.hidden_c, self.hidden_c), 1)
        y = [y[0], y[1]]
        y.extend(block(y[-1]) for block in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV2MerudandaBhag10", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}]"

class VajraV2MerudandaBhag11(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=True, expansion_ratio=0.5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, 1, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(*(InnerBlock(hidden_c, hidden_c, 2, True, 1, 0.5), MerudandaDW(hidden_c, hidden_c, True, 0.5, True, 7, 1))) for _ in range(num_blocks)
        )
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(block(y[-1]) for block in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV2MerudandaBhag11", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}]"

class VajraV2MerudandaBhag12(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, num_blocks=3, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.bottleneck_blocks = nn.ModuleList(
            nn.Sequential(*(InnerBlock(hidden_c, hidden_c, 2, True, 1, 0.5), MerudandaDW(hidden_c, hidden_c, True, 0.5, True, 5, 1))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV2MerudandaBhag12", f"[{self.in_c}, {self.hidden_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV2MerudandaBhag13(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, inner_block=False, block_type="bottleneck", mlp_ratio=1.5):
        super().__init__()
        self.in_c = in_c
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.inner_block = inner_block
        self.expansion_ratio = expansion_ratio
        self.conv1 = ConvBNAct(self.in_c, 3 * self.hidden_c, 1, 1)
        self.block_type = block_type
        self.mlp_ratio = mlp_ratio
        if block_type == "bottleneck":
            self.blocks = nn.ModuleList(InnerBlock(self.hidden_c, self.hidden_c, 2, True, 1) for _ in range(num_blocks)) if inner_block else nn.ModuleList(Bottleneck(self.hidden_c, self.hidden_c, True, expansion_ratio=1.0) for _ in range(num_blocks))
        else:
            self.blocks = nn.ModuleList(
                nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64, kernel_size=7, shortcut=True, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, self.hidden_c, 1, 1)
        self.conv3 = ConvBNAct(2 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        a, b, c = self.conv1(x).chunk(3, 1)
        y = [b, c]
        y.extend(block(y[-1]) for block in self.blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        return self.conv3(torch.cat((a, conv2), 1))
    
    def forward_split(self, x):
        a, b, c = self.conv1(x).split(self.hidden_c, 1)
        y = [b, c]
        y.extend(block(y[-1]) for block in self.blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        return self.conv3(torch.cat((a, conv2), 1))
    
    def get_module_info(self):
        return "VajraV2MerudandaBhag13", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.inner_block}, {self.block_type}, {self.mlp_ratio}]"

class VajraV2MerudandaBhag14(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, inner_block=False, block_type="bottleneck", mlp_ratio=1.5):
        super().__init__()
        self.in_c = in_c
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.inner_block = inner_block
        self.expansion_ratio = expansion_ratio
        self.conv1 = ConvBNAct(self.in_c, 3 * self.hidden_c, 1, 1) if block_type != "attention" else SPPF(self.in_c, 3 * self.hidden_c, 5)
        self.block_type = block_type
        self.mlp_ratio = mlp_ratio
        assert in_c == 2 * hidden_c, "in_c must be equal to 2 * hidden_c for fusion"
        if block_type == "bottleneck":
            self.blocks = nn.ModuleList(InnerBlock(self.hidden_c, self.hidden_c, 2, True, 1) for _ in range(num_blocks)) if inner_block else nn.ModuleList(Bottleneck(self.hidden_c, self.hidden_c, True, expansion_ratio=1.0) for _ in range(num_blocks))
        else:
            self.blocks = nn.ModuleList(
                nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64, kernel_size=7, shortcut=True, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, self.hidden_c, 1, 1)
        self.conv3 = ConvBNAct(2 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        a, b, c = self.conv1(x).chunk(3, 1)
        y = [b, c]
        y.extend(block(y[-1]) for block in self.blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        fm = torch.cat((a, conv2), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        return self.conv3(fm)
    
    def forward_split(self, x):
        a, b, c = self.conv1(x).split(self.hidden_c, 1)
        y = [b, c]
        y.extend(block(y[-1]) for block in self.blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        fm = torch.cat((a, conv2), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        return self.conv3(fm)
    
    def get_module_info(self):
        return "VajraV2MerudandaBhag14", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.inner_block}, {self.block_type}, {self.mlp_ratio}]"

class VajraV2MerudandaBhag15(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        assert in_c == 2 * hidden_c, "in_c must be equal to 2 * hidden_c for fusion"
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        block = VajraV2InnerBlock if inner_block else Bottleneck
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))

    def get_module_info(self):
        return f"VajraV2MerudandaBhag15", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV2MerudandaBhag3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio=0.5, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        block = InnerBlock if inner_block else Bottleneck
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv3  = ConvBNAct((num_blocks + 1) * hidden_c, out_c, 1, 1)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, 2, True, 1, 0.5, False) for _ in range(num_blocks)) if block == InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks))

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        y = [a]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv3(torch.cat((*y[1:], b), dim=1))
        return y
    
    def get_module_info(self):
        return f"VajraV2MerudandaBhag3", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.inner_block}]"

class VajraLiteMerudandaBhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=3, shortcut=False, expand_channels=256, use_cbam = False, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.use_cbam = use_cbam
        self.expand_channels = expand_channels
        self.inner_block = inner_block
        self.kernel_size = kernel_size
        block = InnerBlockLite if inner_block else VajraV1LiteBlock
        self.conv1 = ConvBNAct(in_c, expand_channels, 1, 1)
        self.bottleneck_blocks = nn.ModuleList(block(expand_channels, expand_channels, 2, True, kernel_size, 0.5) for _ in range(num_blocks)) if inner_block else nn.ModuleList(block(expand_channels, expansion_ratio=0.5, kernel_size=kernel_size) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * expand_channels, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return x + cbam if self.add else y + cbam

    def get_module_info(self):
        return f"VajraLiteMerudandaBhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.kernel_size}, {self.shortcut}, {self.expand_channels}, {self.use_cbam}, {self.inner_block}]"

class VajraV2MerudandaBhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expand_ratio=0.5, use_cbam = False, inner_block=False, rep_vgg_k=5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.shortcut = shortcut
        self.use_cbam = use_cbam
        self.hidden_c = int(out_c * expand_ratio)
        self.inner_block = inner_block
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.rep_vgg_k = rep_vgg_k
        block = InnerBlockV2 if inner_block else VajraV2Block
        self.conv1 = ConvBNAct(in_c, self.hidden_c, 1, 1)
        self.bottleneck_blocks = nn.ModuleList(block(self.hidden_c, self.hidden_c, 2, True, kernel_size, 0.5, rep_vgg_k) for _ in range(num_blocks)) if inner_block else nn.ModuleList(block(self.hidden_c, expansion_ratio=0.5, rep_vgg_k=rep_vgg_k) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * self.hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return x + cbam if self.add else y + cbam

    def get_module_info(self):
        return f"VajraV2MerudandaBhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.kernel_size}, {self.shortcut}, {self.expand_ratio}, {self.use_cbam}, {self.inner_block}, {self.rep_vgg_k}]"

class VajraGrivaBhag3(nn.Module):
    def __init__(self, out_c, num_blocks=3, shortcut=False, kernel_size=1, expansion_ratio=0.5, use_cbam=False, inner_block=False) -> None:
        super().__init__()
        block = InnerBlock if inner_block else Bottleneck
        hidden_c = int(out_c * expansion_ratio)
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.use_cbam = use_cbam
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, 2, True, 1, 0.5) for _ in range(num_blocks)) if block == InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) #nn.ModuleList(block(hidden_c, hidden_c, 2, True, kernel_size=kernel_size, expansion_ratio=0.5, bhag1=bhag1) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()

    def forward(self, x):
        y = [x]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return y + cbam

    def get_module_info(self):
        return f"VajraGrivaBhag3", f"[{self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.use_cbam}]"    

class VajraV2MerudandaBhag1(nn.Module):
    def __init__(self, in_c, mid_c1, mid_c2, out_c, num_bottleneck_blocks=2) -> None:
        super().__init__()
        self.in_c = in_c
        self.mid_c1 = mid_c1
        self.mid_c2 = mid_c2
        self.out_c = out_c
        self.num_blocks = num_bottleneck_blocks
        self.hidden_c = mid_c1 // 2
        self.conv1 = ConvBNAct(in_c, mid_c1, 1, 1)
        self.block1 = InnerBlock(mid_c1 // 2, mid_c2, num_bottleneck_blocks, True)
        self.block2 = InnerBlock(mid_c2, mid_c2, num_bottleneck_blocks, True)
        self.conv2 = ConvBNAct(mid_c1 + (2 * mid_c2), out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend((block(y[-1])) for block in [self.block1, self.block2])
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.conv1(x).split((self.hidden_c, self.hidden_c), 1))
        y.extend(block(y[-1]) for block in [self.block1, self.block2])
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV2MerudandaBhag1", f"[{self.in_c}, {self.mid_c1}, {self.mid_c2}, {self.out_c}, {self.num_blocks}]"

class VajraMerudandaBhag4(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, expansion_ratio=0.5, inner_block=False, num_bottleneck_blocks=2, use_cbam=False) -> None:
        super().__init__()
        block = InnerBlock if inner_block else Bottleneck
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.inner_block = inner_block
        self.use_cbam = use_cbam
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c // 2, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, num_bottleneck_blocks, shortcut=True, kernel_size=1, expansion_ratio=0.5) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c // 2 + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        a, b = x.chunk(2, 1)
        y = [a, self.conv1(b)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return x + cbam if self.add else y + cbam
    
    def forward_split(self, x):
        a, b = x.split((self.in_c // 2, self.in_c // 2), 1)
        y = [a, self.conv1(b)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return x + cbam if self.add else y + cbam

    def get_module_info(self):
        return f"VajraMerudandaBhag4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.inner_block}, {self.use_cbam}]"

class VajraGrivaBhag4(nn.Module):
    def __init__(self, out_c, num_blocks=3, shortcut=False, kernel_size=1, expansion_ratio=0.5, bhag1=False, use_cbam=False, bottleneck_dw=False) -> None:
        super().__init__()
        block = VajraMerudandaBhag1 if bhag1 else Bottleneck
        hidden_c = int(out_c * expansion_ratio)
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.bhag1 = bhag1
        self.use_cbam = use_cbam
        self.kernel_size=kernel_size
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, 2, shortcut=True, kernel_size=1, expansion_ratio=0.5, bottleneck_dwcib=bottleneck_dw) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()

    def forward(self, x):
        y = [x]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return y + cbam

    def get_module_info(self):
        return f"VajraGrivaBhag4", f"[{self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.bhag1}, {self.use_cbam}]"

class VajraMerudandaBhag5(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, expansion_ratio=0.5, inner_block=False, num_bottleneck_blocks=2, use_cbam=False) -> None:
        super().__init__()
        block = InnerBlockV2 if inner_block else BottleneckV2
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.inner_block = inner_block
        self.use_cbam = use_cbam
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, expansion_ratio=0.5) for _ in range(num_blocks)) if block == BottleneckV2 else nn.ModuleList(block(hidden_c, hidden_c, num_bottleneck_blocks, kernel_size=1) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return y

    def get_module_info(self):
        return f"VajraMerudandaBhag5", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.inner_block}, {self.use_cbam}]"
    
class VajraMerudandaBhag6(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut = shortcut
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        hidden_c = int(out_c * expansion_ratio)
        block = Bottleneck
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, 1)
        self.conv_branch = ConvBNAct(hidden_c, hidden_c, 1, 3)
        self.branch_b_bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 3) * hidden_c, out_c, 1, 1)
        self.cbam = CBAM(out_c)
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        branch_conv = self.conv_branch(a)
        y = [x, a, branch_conv, b]
        y.extend(branch_b_bottleneck_block(y[-1]) for branch_b_bottleneck_block in self.branch_b_bottleneck_blocks)
        conv2 = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(conv2)
        return x + cbam if self.add else conv2 + cbam

    def get_module_info(self):
        return f"VajraMerudandaBhag6", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}]"

class VajraMerudandaBhag7(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, expansion_ratio=0.5, inner_block=False, num_bottleneck_blocks=2, use_cbam=False) -> None:
        super().__init__()
        block = InnerBlock if inner_block else Bottleneck
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.inner_block = inner_block
        self.use_cbam = use_cbam
        self.kernel_size=kernel_size
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, num_bottleneck_blocks, shortcut=True, kernel_size=1, expansion_ratio=0.5) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        #a, b = x.chunk(2, 1)
        #y = [a, self.conv1(b)]
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return y
    
    def forward_split(self, x):
        #a, b = x.split((self.in_c // 2, self.in_c // 2), 1)
        #y = [a, self.conv1(b)]
        y = list(self.conv1(x).split((self.hidden_c, self.hidden_c), 1))
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return y

    def get_module_info(self):
        return f"VajraMerudandaBhag7", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.inner_block}, {self.use_cbam}]"

class VajraGrivaBhag1(nn.Module):
    def __init__(self, out_c, num_blocks=3, kernel_size=1, expansion_ratio=0.5, use_cbam=False, bottleneck_dw=False, use_rep_vgg_dw=False) -> None:
        super().__init__()
        block = Bottleneck if not bottleneck_dw else MerudandaDW
        hidden_c = int(out_c * expansion_ratio)
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.kernel_size=kernel_size
        self.use_cbam = use_cbam
        self.bottleneck_dw = bottleneck_dw
        self.use_rep_vgg_dw = use_rep_vgg_dw
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=True, expansion_ratio=1.0) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, shortcut=True, expansion_ratio=1.0, use_rep_vgg_dw=use_rep_vgg_dw) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()

    def forward(self, x):
        y = [x]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return y + cbam

    def get_module_info(self):
        return f"VajraGrivaBhag1", f"[{self.out_c}, {self.num_blocks}, {self.kernel_size}, {self.expansion_ratio}, {self.use_cbam}, {self.bottleneck_dw}, {self.use_rep_vgg_dw}]"

class InnerBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, kernel_size=1, expansion_ratio=0.5, bottleneck_dwcib=False) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, stride=1, kernel_size=kernel_size)
        self.conv2 = ConvBNAct(in_c, hidden_c, kernel_size=1, stride=1)
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(*[Bottleneck(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks)]) if not bottleneck_dwcib else nn.Sequential(*(MerudandaDW(hidden_c, hidden_c, True, 0.5, True, 7, 1) for _ in range(2)))
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return out

class InnerBlockMLPResViT(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, expansion_ratio=0.5, res_vit=False) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, stride=1, kernel_size=1)
        self.mlp = nn.Sequential(ConvBNAct(in_c, 2 * hidden_c, 1, 1), ConvBNAct(2 * hidden_c, hidden_c, 1, 1, act=None))
        self.conv2 = ConvBNAct(in_c + hidden_c, out_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(VajraResViTBlock(hidden_c, hidden_c, False), VajraResViTBlock(hidden_c, hidden_c, False))
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        mlp = x + self.mlp(x)
        out = self.conv2(torch.cat((b, mlp), 1))
        return out
    
class InnerBlockResViT(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, expansion_ratio=0.5, res_vit=False) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(*(VajraResViTBlock(hidden_c, hidden_c, False, 1.2) for _ in range(num_blocks)))
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return out
    
class InnerBlockRepViT(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(*(VajraRepViTBlock(hidden_c, hidden_c, use_se=True if i % 2 == 0 else False, kernel_size=7, swiglu_ffn=False, mlp_ratio=2.0, use_rep_vgg_dw=True) for i in range(num_blocks)))
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return out
    
class VajraV1MerudandaBhag9(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, num_bottleneck_blocks=2, bottleneck_expansion_ratio=1.0, rep_bottleneck=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        assert in_c == 2 * hidden_c, "in_c must be equal to 4 * hidden_c for fusion"
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        if inner_block:
            block = VajraV2InnerBlock
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck
        self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size, num_bottleneck_blocks=num_bottleneck_blocks, bottleneck_expansion=bottleneck_expansion_ratio, rep_bottleneck=rep_bottleneck) for _ in range(num_blocks))) if block == VajraV2InnerBlock else nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat(y, 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        #y.append(fm)
        return self.conv2(fm)

    def get_module_info(self):
        return f"VajraV1MerudandaBhag9", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"
    
class VajraV1MerudandaBhag10(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5, inner_block=False, mlp_res_vit=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.mlp_res_vit = mlp_res_vit
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, 1)
        if inner_block:
            if mlp_res_vit:
                block = InnerBlockMLPResViT
                self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, num_blocks=2, shortcut=True, res_vit=True) for _ in range(num_blocks)))
            else:
                block = InnerBlockResViT
                self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, num_blocks=2, shortcut=True, res_vit=True) for _ in range(num_blocks)))
        else:
            block = VajraRepViTBlock
            self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, use_se=False if i % 2 == 0 else True, stride=1, use_rep_vgg_dw=False, kernel_size=3, mlp_ratio=1.2) for i in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag10", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.inner_block}, {self.mlp_res_vit}]"

class VajraV1MerudandaBhag15(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5, use_mlp=False, kernel_size=7, use_repvgg_dw = False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, 1)
        if not use_mlp:
            self.bottleneck_blocks = nn.ModuleList(
                nn.Sequential(MerudandaDW(hidden_c, hidden_c, True, 1.0, use_rep_vgg_dw=use_repvgg_dw, kernel_size=kernel_size, stride=1)) for i in range(num_blocks)
            )
        else:
            self.bottleneck_blocks = nn.ModuleList(
                VajraRepViTBlock(hidden_c, hidden_c, use_se = False, stride=1, use_rep_vgg_dw=use_repvgg_dw, kernel_size=kernel_size, swiglu_ffn=False, mlp_ratio=2.0) for i in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag15", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}]"

class VajraV1MerudandaBhag16(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, num_bottleneck_blocks=4, bottleneck_expansion_ratio=1.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        assert in_c == 2 * hidden_c, "in_c must be equal to 2 * hidden_c for fusion"
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(
            nn.Sequential(InnerBlockRepViT(hidden_c, hidden_c, 4, 0.5), Bottleneck(hidden_c, hidden_c, True, expansion_ratio=1.0)) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag16", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MerudandaBhag11(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, rep_bottleneck=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        if inner_block:
            block = VajraV2InnerBlock
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck

        if block == VajraV2InnerBlock:
            self.bottleneck_blocks = nn.ModuleList(
                nn.Sequential(block(hidden_c, hidden_c, use_attn=use_attn, use_bottleneck_v3=use_bottleneck_v3, shortcut=True, grid_size=grid_size, rep_bottleneck=rep_bottleneck), ConvBNAct(hidden_c, hidden_c, 1, 3)) for _ in range(num_blocks)
            )
        else:
            self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag11", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MerudandaBhag17(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1, shortcut=False, expansion_ratio = 0.5, use_bottleneck_v3 = False, inner_block=False, use_attn=False, grid_size=1, num_bottleneck_blocks=4, bottleneck_expansion_ratio=1.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        self.use_attn = use_attn
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        assert in_c == 2 * hidden_c, "in_c must be equal to 2 * hidden_c for fusion"
        self.conv1 = ConvBNAct(in_c, 2 * hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(
            nn.Sequential(VajraV2InnerBlock(hidden_c, hidden_c, False, False, True, num_bottleneck_blocks=2), InnerBlockRepViT(hidden_c, hidden_c, 4)) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag17", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.use_bottleneck_v3}, {self.inner_block}, {self.use_attn}]"

class VajraV1MerudandaBhag12(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        if inner_block:
            block = InnerBlock
            self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, num_blocks=2, shortcut=True, kernel_size=1) for _ in range(num_blocks)))
        else:
            block = Bottleneck
            self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, True) for _ in range(num_blocks)))
        self.conv3 = ConvBNAct((num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv3(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag12", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.inner_block}]"

class VajraV1MerudandaBhag13(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        if inner_block:
            block = InnerBlock
            self.bottleneck_blocks = nn.Sequential(*(block(hidden_c, hidden_c, num_blocks=2, shortcut=True, kernel_size=1) for _ in range(num_blocks)))
        else:
            block = Bottleneck
            self.bottleneck_blocks = nn.Sequential(*(block(hidden_c, hidden_c, True) for _ in range(num_blocks)))
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(self.conv2(x))
        return self.conv3(torch.cat((a, b), 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag13", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.inner_block}]"

class VajraV1MerudandaBhag14(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5, inner_block=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        self.inner_block = inner_block
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        if inner_block:
            block = InnerBlock
            self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, num_blocks=2, shortcut=True, kernel_size=1) for _ in range(num_blocks)))
        else:
            block = Bottleneck
            self.bottleneck_blocks = nn.ModuleList((block(hidden_c, hidden_c, True) for _ in range(num_blocks)))
        self.conv3 = ConvBNAct(in_c + (num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        y = [x, a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv3(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1MerudandaBhag14", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.expansion_ratio}, {self.inner_block}]"

class InnerBlockMakkarNorm(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, kernel_size=1, expansion_ratio=0.5, bottleneck_dwcib=False) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvMakkarNorm(in_c, hidden_c, stride=1, kernel_size=kernel_size)
        self.conv2 = ConvMakkarNorm(in_c, hidden_c, kernel_size=1, stride=1)
        self.conv3 = ConvMakkarNorm(2 * hidden_c, out_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(*[BottleneckMakkarNorm(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=1.0) for _ in range(num_blocks)]) if not bottleneck_dwcib else nn.Sequential(*[MerudandaDW(hidden_c, hidden_c, True, 0.5, True, 7, 1) for _ in range(num_blocks)])
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return out 
    
class InnerBlockAttention(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, kernel_size=1, expansion_ratio=0.5, lite=False) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, stride=1, kernel_size=kernel_size)
        self.conv2 = ConvBNAct(in_c, hidden_c, kernel_size=1, stride=1)
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, 1, 1)
        self.add = in_c == out_c
        self.attn_blocks = nn.Sequential(*[AttentionBlock(hidden_c, hidden_c, num_heads = hidden_c // 32 if not lite else hidden_c // 8, shortcut=shortcut) for _ in range(num_blocks)])
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.attn_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return out
    
class InnerBlockLite(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, kernel_size=3, expansion_ratio=0.5) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, stride=1, kernel_size=1)
        self.conv2 = ConvBNAct(2 * hidden_c, out_c, kernel_size=1, stride=1)
        self.bottleneck_blocks = nn.Sequential(*[VajraV1LiteBlock(hidden_c, expansion_ratio=1.0, kernel_size=kernel_size) for _ in range(num_blocks)])
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv2(torch.cat((a, b), 1))
        return x + out if self.add else out
    
class VajraV1LiteOuterBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, kernel_size=3, expand_channels=256, use_cbam=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.expand_channels = expand_channels
        self.use_cbam = use_cbam
        self.conv1 = ConvBNAct(in_c, expand_channels, 1, 1)
        self.bottleneck_blocks = nn.Sequential(*[VajraV1LiteBlock(expand_channels, expansion_ratio=1.0, kernel_size=kernel_size) for _ in range(num_blocks)])
        self.conv2 = ConvBNAct(expand_channels, out_c, 1, 1)
        self.cbam = CBAM(out_c) if use_cbam else nn.Identity()
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        conv2 = self.conv2(b)
        cbam = self.cbam(conv2)
        return x + cbam if self.add else cbam + conv2
    
    def get_module_info(self):
        return "VajraV1LiteOuterBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expand_channels}, {self.use_cbam}]"
    
class InnerBlockV2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, shortcut=False, kernel_size=1, expansion_ratio=0.5, rep_vgg_k=5) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, stride=1, kernel_size=kernel_size)
        self.conv2 = ConvBNAct(2 * hidden_c, out_c, kernel_size=1, stride=1)
        self.bottleneck_blocks = nn.Sequential(*[VajraV2Block(hidden_c, expansion_ratio=1.0, rep_vgg_k=rep_vgg_k) for _ in range(num_blocks)])
        self.add = shortcut and in_c == out_c
    
    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv2(torch.cat((a, b), 1))
        return x + out if self.add else out
    
class VajraMerudandaBhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, expansion_ratio=0.5, bhag1=False, use_cbam=False, bottleneck_dw=False) -> None:
        super().__init__()
        block = VajraMerudandaBhag1 if bhag1 else Bottleneck
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.expansion_ratio = expansion_ratio
        self.num_blocks=num_blocks
        self.shortcut=shortcut
        self.bhag1 = bhag1
        self.use_cbam = use_cbam
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, 2, shortcut=True, kernel_size=1, expansion_ratio=0.5, bottleneck_dwcib=bottleneck_dw) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return x + cbam if self.add else y + cbam

    def get_module_info(self):
        return f"VajraMerudandaBhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.expansion_ratio}, {self.bhag1}, {self.use_cbam}]"

class VajraGrivaBhag2(nn.Module):
    def __init__(self, out_c, num_blocks=3, kernel_size=1, num_bottleneck_blocks=1, bhag1=False) -> None:
        super().__init__()
        block = VajraMerudandaBhag1 if bhag1 else Bottleneck
        hidden_c = int(out_c * 0.5)
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.kernel_size=kernel_size
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, shortcut=True, expansion_ratio=0.5, kernel_size=(3, 3)) for _ in range(num_blocks)) if block == Bottleneck else nn.ModuleList(block(hidden_c, hidden_c, num_bottleneck_blocks, True, 1, False, 0.5) for _ in range(num_blocks)) #nn.ModuleList(block(hidden_c, hidden_c, shortcut=True, kernel_size=kernel_size, num_blocks=num_bottleneck_blocks, bottleneck_dwcib=bottleneck_dw) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        y = [x]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y = self.conv2(torch.cat(y, 1))
        return y

    def get_module_info(self):
        return f"VajraGrivaBhag2", f"[{self.out_c}, {self.num_blocks}, {self.kernel_size}]"

class VajraAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, embed_channels=128, num_heads=1, guide_channels=512, inner_block=False, use_cbam = False) -> None:
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
        self.inner_block = inner_block
        self.use_cbam = use_cbam
        block = InnerBlock if inner_block else Bottleneck
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, 2, True, 1, 0.5) for _ in range(num_blocks)) if block == InnerBlock else nn.ModuleList(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=0.5) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 2) * hidden_c, out_c, kernel_size=1, stride=1)
        self.add = shortcut and in_c == out_c
        self.cbam = CBAM(out_c) if self.use_cbam else nn.Identity()
        self.attn = MaxSigmoidAttentionBlock(hidden_c, hidden_c, num_heads, embed_channels, guide_channels)

    def forward(self, x, guide):
        y = [x, self.conv1(x)]
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_blocks)
        y.append(self.attn(y[-1], guide))
        y = self.conv2(torch.cat(y, 1))
        cbam = self.cbam(y)
        return cbam + x if self.add else cbam + y

    def get_module_info(self):
        return f"VajraAttentionBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.embed_channels}, {self.num_heads}, {self.guide_channels}, {self.inner_block}, {self.use_cbam}]"

class VajraV2BottleneckBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, num_bottleneck_blocks=2, shortcut=False, kernel_size=1, bottleneck_kernel_size=3) -> None:
        super().__init__()
        block = VajraMerudandaBhag1
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.bottleneck_kernel_size = bottleneck_kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, num_blocks=num_bottleneck_blocks, shortcut=shortcut, kernel_size=bottleneck_kernel_size) for _ in range(num_blocks))
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
        return f"VajraV2BottleneckBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.num_bottleneck_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_kernel_size}]"

class VajraV2InnerBlock(nn.Module):
    def __init__(self, in_c, out_c, use_attn = False, use_bottleneck_v3=False, shortcut=False, grid_size=1, num_bottleneck_blocks=2, bottleneck_expansion=1.0, rep_bottleneck=True) -> None:
        super().__init__()
        #block = AreaAttentionBlock if use_attn else RepBottleneck
        if use_attn:
            block = AreaAttentionBlock
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut = shortcut
        self.use_attn = use_attn
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.hidden_c = hidden_c
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        if use_attn:
            self.gamma = nn.Parameter(0.01 * torch.ones(out_c), requires_grad=True)
            self.bottleneck_blocks = nn.Sequential(*([block(hidden_c, hidden_c, hidden_c // 32, 7, True, area=grid_size)] * 2))
        elif use_bottleneck_v3:
            self.bottleneck_blocks = nn.Sequential(*([BottleneckV3(hidden_c, hidden_c, True, 5, 1.0), BottleneckV3(hidden_c, hidden_c, True, 5, 1.0)]))
        else:
            self.bottleneck_blocks = nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=bottleneck_expansion) for _ in range(num_bottleneck_blocks)))
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return x + self.gamma.view(-1, len(self.gamma), 1, 1) * out if self.use_attn else out

    def get_module_info(self):
        return f"VajraV2InnerBlock", f"[{self.in_c}, {self.out_c}, {self.use_attn}, {self.use_bottleneck_v3}, {self.shortcut}]"
    
class VajraV3InnerBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, kernel_size=1, bottleneck_pyconv=False) -> None:
        super().__init__()
        block = BottleneckV3
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
        return f"VajraV3InnerBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_pyconv}]"

class VajraV3MidBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, num_bottleneck_blocks=2, shortcut=False, kernel_size=1, bottleneck_kernel_size=3) -> None:
        super().__init__()
        block = VajraV3InnerBlock
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.bottleneck_kernel_size = bottleneck_kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, num_blocks=num_bottleneck_blocks, shortcut=shortcut, kernel_size=bottleneck_kernel_size) for _ in range(num_blocks))
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
        return f"VajraV2MidBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.num_bottleneck_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_kernel_size}]"

class VajraV3BottleneckBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, num_bottleneck_blocks=3, shortcut=False, kernel_size=1, bottleneck_kernel_size=1) -> None:
        super().__init__()
        block = VajraV3MidBlock
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks=num_blocks
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.shortcut=shortcut
        self.kernel_size=kernel_size
        self.bottleneck_kernel_size = bottleneck_kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size)
        self.bottleneck_blocks = nn.ModuleList(block(hidden_c, hidden_c, num_blocks=1, num_bottleneck_blocks=num_bottleneck_blocks, shortcut=shortcut, kernel_size=3, bottleneck_kernel_size=bottleneck_kernel_size) for _ in range(num_blocks))
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
        return f"VajraV3BottleneckBlock", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.num_bottleneck_blocks}, {self.shortcut}, {self.kernel_size}, {self.bottleneck_kernel_size}]"

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
        self.vajra_bottleneck = VajraMerudandaBhag1(in_c=hidden_c, out_c=hidden_c, num_blocks=1, shortcut=True, kernel_size=3) #VajraV2BottleneckBlock(in_c=hidden_c, out_c=hidden_c, num_blocks=1, num_bottleneck_blocks=1, shortcut=True, kernel_size=3) #
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

class VajraV3StemBlock(nn.Module):
    def __init__(self, in_c, hidden_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.hidden_c = hidden_c
        self.out_c = out_c
        self.conv1 = ConvBNAct(in_c, hidden_c, kernel_size=3, stride=2)
        self.vajra_bottleneck = VajraV2BottleneckBlock(in_c=hidden_c, out_c=hidden_c, num_blocks=1, num_bottleneck_blocks=1, shortcut=True, kernel_size=3) #VajraV3InnerBlock(in_c=hidden_c, out_c=hidden_c, num_blocks=1, shortcut=True, kernel_size=3)
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
        return f"VajraV3StemBlock", f"[{self.in_c}, {self.hidden_c}, {self.out_c}]"

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

class C3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, shortcut=False, expansion_ratio=0.5, kernel_size=(1, 3)) -> None:
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.num_blocks = num_blocks
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut = shortcut
        self.add = shortcut and in_c == out_c

        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv3 = ConvBNAct(2 * hidden_c, out_c, 1, 1)

        self.bottleneck_blocks = nn.Sequential(*(Bottleneck(hidden_c, hidden_c, shortcut=shortcut, kernel_size=kernel_size, expansion_ratio=1.0) for _ in range(num_blocks)))

    def forward(self, x):
        return self.conv3(torch.cat((self.bottleneck_blocks(self.conv1(x)), self.conv2(x)), 1))

class MaxSigmoidAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=1, embed_channels=512, guide_channels=512, scale=False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_channels = out_c // num_heads
        self.embed_conv = ConvBNAct(in_c, out_c, kernel_size=1, act=None) if in_c != embed_channels else None
        self.guide_linear = nn.Linear(guide_channels, embed_channels)
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
    
class DvayaSanlayan(nn.Module):
    def __init__(self, in_c, out_c, use_cbam=False, expansion_ratio=1.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        total_c = sum(in_c)
        self.expansion_ratio = expansion_ratio
        self.out_channels = int(out_c * expansion_ratio)
        self.conv_fused = ConvBNAct(total_c, self.out_channels, 1, 1)
        self.use_cbam = use_cbam
        self.cbam = CBAM(self.out_channels) if self.use_cbam else nn.Identity()

    def forward(self, x):
        _, _, H, W = x[1].shape
        concat_inp = torch.cat((x), dim=1)
        fused = self.conv_fused(concat_inp)
        cbam = self.cbam(fused)
        return cbam + fused if self.use_cbam else fused
    
    def get_module_info(self):
        return "DvayaSanlayan", f"[{self.in_c}, {self.out_c}, {self.use_cbam}, {self.expansion_ratio}]"
    
class TritayaSanlayan(nn.Module):
    def __init__(self, in_c, out_c, use_cbam=False, expansion_ratio=1.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        total_c = sum(in_c)
        self.expansion_ratio = expansion_ratio
        self.out_channels = int(out_c * expansion_ratio)
        self.conv_fused = ConvBNAct(total_c, self.out_channels, 1, 1)
        self.use_cbam = use_cbam
        self.cbam = CBAM(self.out_channels) if self.use_cbam else nn.Identity()

    def forward(self, x):
        _, _, H, W = x[1].shape
        x0 = F.interpolate(x[0], size=(H, W), mode="nearest")
        x1 = x[1]
        x2 = F.interpolate(x[2], size=(H, W), mode='nearest')
        concatenated_oup = torch.cat((x0, x1, x2), dim=1)
        fused = self.conv_fused(concatenated_oup)
        cbam = self.cbam(fused)
        return fused + cbam if self.use_cbam else fused

    def get_module_info(self):
        return f"TritayaSanlayan", f"[{self.in_c}, {self.out_c}, {self.use_cbam}, {self.expansion_ratio}]"

class ChatushtayaSanlayan(nn.Module):
    def __init__(self, in_c, out_c, use_cbam=True, expansion_ratio=1.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        total_c = sum(in_c)
        self.expansion_ratio = expansion_ratio
        self.out_channels = int(out_c * expansion_ratio)
        self.conv_fused = ConvBNAct(total_c, self.out_channels, 1, 1)
        self.use_cbam = use_cbam
        self.cbam = CBAM(self.out_channels) if self.use_cbam else nn.Identity()

    def forward(self, x):
        _, _, H, W = x[2].shape
        x0 = F.interpolate(x[0], size=(H, W), mode="nearest")
        x1 = F.interpolate(x[1], size=(H, W), mode="nearest")
        x2 = x[2]
        x3 = F.interpolate(x[3], size=(H, W), mode='nearest')
        concatenated_oup = torch.cat((x0, x1, x2, x3), dim=1)
        fused = self.conv_fused(concatenated_oup)
        cbam = self.cbam(fused)
        return fused + cbam if self.use_cbam else fused

    def get_module_info(self):
        return f"ChatushtayaSanlayan", f"[{self.in_c}, {self.out_c}, {self.use_cbam}, {self.expansion_ratio}]"

class Sanlayan(nn.Module):
    def __init__(self, in_c, out_c, stride=2, use_cbam=False, expansion_ratio=1.0, kernel_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        total_c = sum(in_c)
        self.use_cbam = use_cbam
        self.expansion_ratio = expansion_ratio
        self.out_channels = int(out_c * expansion_ratio)
        self.conv_fused = ConvBNAct(total_c, self.out_channels, 1, kernel_size=kernel_size)
    
    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        out = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs] #[adaptive_pool(inp) for inp in inputs]
        concatenated_out = torch.cat(out, dim=1)
        conv = self.conv_fused(concatenated_out)
        return conv

    def get_module_info(self):
        return f"Sanlayan", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.use_cbam}]"

class VajraStambh(nn.Module):
    """ Inspired by the Stem Block of PPHGNetV2 """
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.stem1 = ConvBNAct(in_c, mid_c, kernel_size=3, stride=2, act="silu")
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=3, stride=2, act="silu")
        self.stem4 = ConvBNAct(mid_c, out_c, kernel_size=1, stride=1, act="silu")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=False)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def get_module_info(self):
        return f"VajraStambh", f"[{self.in_c}, {self.mid_c}, {self.out_c}]"
    
class VajraV2Stambh(nn.Module):
    """ Inspired by the Stem Block of PPHGNetV2 """
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.stem1 = ConvMakkarNorm(in_c, mid_c, kernel_size=3, stride=2)
        self.stem2a = ConvMakkarNorm(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0)
        self.stem2b = ConvMakkarNorm(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0)
        self.stem3 = ConvMakkarNorm(mid_c * 2, mid_c, kernel_size=3, stride=2)
        self.stem4 = ConvMakkarNorm(mid_c, out_c, kernel_size=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=False)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def get_module_info(self):
        return f"VajraV2Stambh", f"[{self.in_c}, {self.mid_c}, {self.out_c}]"
    
class VajraDownsample(nn.Module):
    """ Inspired by the Stem Block of PPHGNetV2 """
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.conv1 = DepthwiseConvBNAct(in_c, mid_c, 1, 3)
        self.pool = MaxPool(2, 2)
        self.conv2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0, act="silu")
        self.conv2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0, act="silu")
        self.conv3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=1, stride=1, act="silu")
        self.conv4 = DepthwiseConvBNAct(mid_c, out_c, 1, 3)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x2 = self.conv2a(x)
        x2 = self.pad(x2)
        x2 = self.conv2b(x2)
        x2 = self.pad(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def get_module_info(self):
        return f"VajraDownsample", f"[{self.in_c}, {self.mid_c}, {self.out_c}]"
    
class VajraDownsampleV2(nn.Module):
    """ Inspired by the Stem Block of PPHGNetV2, modified for total stride of 2 """
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.stem1 = ConvBNAct(in_c, mid_c, kernel_size=3, stride=2, act="silu")
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=3, stride=1, padding=1, act="silu")
        self.stem4 = ConvBNAct(mid_c, out_c, kernel_size=1, stride=1, act="silu")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=False)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def get_module_info(self):
        return f"VajraDownsampleV2", f"[{self.in_c}, {self.mid_c}, {self.out_c}]"
    
class RepVGGDW_3x3(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1) -> None:
        super().__init__()
        self.kernel_size=kernel_size
        self.stride = stride
        self.conv = DepthwiseConvBNAct(dim, dim, stride=stride, kernel_size=kernel_size, act=None)
        self.conv1 = ConvBNAct(dim, dim, 1, 1, act=None)
        self.dim = dim
        self.act = nn.SiLU()
        self.fused_conv = None  # Placeholder for fused layer

    def forward(self, x):
        if self.fused_conv is not None:
            return self.act(self.fused_conv(x))
        return self.act(self.conv(x) + self.conv1(x))
    
    def _calculate_padding(self, kernel, target_size):
        """Calculate padding required to resize the kernel to the target size."""
        current_size = kernel.shape[2]
        pad_total = target_size - current_size
        pad = pad_total // 2
        return [pad, pad + (pad_total % 2), pad, pad + (pad_total % 2)]
    
    @torch.no_grad()
    def fuse(self):
        # Fuse conv and batchnorm layers for self.conv and self.conv1
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        # Dynamically adjust padding for any kernel size
        if conv_w.shape[2:] != conv1_w.shape[2:]:
            larger_size = max(conv_w.shape[2], conv1_w.shape[2])
            conv_w = torch.nn.functional.pad(conv_w, self._calculate_padding(conv_w, larger_size))
            conv1_w = torch.nn.functional.pad(conv1_w, self._calculate_padding(conv1_w, larger_size))

        # Combine the two weights and biases
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        # Create a new fused_conv layer with combined weights and biases
        device = next(self.parameters()).device
        self.fused_conv = nn.Conv2d(
            in_channels=self.conv.conv.in_channels,
            out_channels=self.conv.conv.out_channels,
            kernel_size=self.conv.conv.kernel_size,
            stride=self.conv.conv.stride,
            padding=self.conv.conv.padding,
            groups=self.conv.conv.groups,
            bias=True
        ).to(device)
        
        self.fused_conv.weight.data.copy_(final_conv_w.to(device))
        self.fused_conv.bias.data.copy_(final_conv_b.to(device))

    def forward_fuse(self, x):
        return self.act(self.fused_conv(x)) if self.fused_conv is not None else self.act(self.conv(x) + self.conv1(x))

    def get_module_info(self):
        return "RepVGGDW_3x3", f"[{self.dim}, {self.kernel_size}, {self.stride}]"

class VajraStambhV2(nn.Module):
    """ For downsampling by stride of 4 """
    """def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden_c = out_c // 2
        self.conv1 = ConvBNAct(in_c, 2 * self.hidden_c, 2, 3)
        self.conv2 = ConvBNAct(self.hidden_c, self.hidden_c, 2, 3)
        self.conv3 = ConvBNAct(2 * self.hidden_c, self.out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x) #torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.conv2(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        out = self.conv3(torch.cat((x1, x2), 1))
        return out"""
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.stem1 = ConvBNAct(in_c, mid_c, kernel_size=3, stride=2, act="silu")
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=3, stride=2, act="silu")
        self.stem4 = ConvBNAct(mid_c, out_c, kernel_size=1, stride=1, act="silu")
        self.token_mixer = nn.Sequential(
            RepVGGDW_3x3(mid_c, kernel_size=3),
            SELayer(mid_c, 4),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=False)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        tm = self.token_mixer(x)
        x = self.stem4(tm)
        return x
    
    def get_module_info(self):
        return "VajraStambhV2", f"[{self.in_c}, {self.out_c}]"
    
class VajraStambhV3(nn.Module):
    """ For downsampling by stride of 4 """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden_c = out_c // 2
        #self.conv1 = ConvBNAct(in_c, self.hidden_c // 2, 2, 3)
        #self.conv2 = ConvBNAct(self.hidden_c // 2, self.hidden_c, 1, 3)
        self.token_mixer1 = nn.Sequential(
            ConvBNAct(in_c, self.hidden_c, 2, 3),
            SELayer(self.hidden_c, 4),
            ConvBNAct(self.hidden_c, self.hidden_c, 1, 1)
        )
        self.channel_mixer1 = nn.Sequential(
            ConvBNAct(self.hidden_c, 2*self.hidden_c, 1, 1),
            ConvBNAct(2*self.hidden_c, self.hidden_c, 1, 1, act=None)
        )
        self.token_mixer2 = nn.Sequential(
            ConvBNAct(self.hidden_c, out_c, 2, 3),
            SELayer(out_c, 4),
            ConvBNAct(out_c, out_c, 1, 1)
        )
        self.channel_mixer2 = nn.Sequential(
            ConvBNAct(out_c, 2*out_c, 1, 1),
            ConvBNAct(2*out_c, out_c, 1, 1, act=None)
        )

    def forward(self, x):
        tm1 = self.token_mixer1(x)
        cm1 = self.channel_mixer1(tm1)
        cm1 = tm1 + cm1
        tm2 = self.token_mixer2(cm1)
        cm2 = self.channel_mixer2(tm2)
        return tm2 + cm2
    
    def get_module_info(self):
        return "VajraStambhV3", f"[{self.in_c}, {self.out_c}]"

class VajraStambhV4(nn.Module):
    """ Inspired by the Stem Block of PPHGNetV2 """
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.stem1 = ConvBNAct(in_c, mid_c, kernel_size=3, stride=2, act="silu")
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0, act="silu")
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=3, stride=2, groups=2, act="silu")
        self.stem4 = ConvBNAct(mid_c, out_c, kernel_size=1, stride=1, act="silu")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=False)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def get_module_info(self):
        return f"VajraStambhV4", f"[{self.in_c}, {self.mid_c}, {self.out_c}]"

class VajraV2DownsampleStem(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c
        self.stem1 = ConvBNAct(in_c, mid_c, kernel_size=3, stride=2, act="relu")
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, kernel_size=2, stride=1, padding=0, act="relu")
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, kernel_size=2, stride=1, padding=0, act="relu")
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, kernel_size=3, stride=2, act="relu")
        self.stem4 = ConvBNAct(mid_c, out_c, kernel_size=1, stride=1, act="relu")
        self.vajra_bottleneck = VajraMerudandaBhag1(in_c=mid_c, out_c=mid_c, num_blocks=1, shortcut=True, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=False)

        self.pad1 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.stem1(x)
        x = self.pad1(x)
        x2 = self.stem2a(x)
        x2 = self.pad2(x2)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x2 = self.vajra_bottleneck(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def get_module_info(self):
        return f"VajraV2DownsampleStem", f"[{self.in_c}, {self.mid_c}, {self.out_c}]"

class VajraSPPModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size=kernel_size
        hidden_dim = in_c // 2
        self.hidden_c = hidden_dim
        self.conv1 = ConvBNAct(in_c, 2*hidden_dim, 1, 1)
        self.conv2 = ConvBNAct(hidden_dim * 5, out_c, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [a, b]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return "VajraSPPModule", f"[{self.in_c}, {self.out_c}, {self.kernel_size}]"

    """def __init__(self, in_c, out_c, kernel_size=5):
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
        return final + z"""

class SPPF(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size=kernel_size
        hidden_dim = in_c // 2
        self.conv1 = ConvBNAct(in_c, hidden_dim, 1, 1)
        self.conv2 = ConvBNAct(hidden_dim * 4, out_c, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "SPPF", f"[{self.in_c}, {self.out_c}, {self.kernel_size}]"

class SPPELAN(nn.Module):
    def __init__(self, in_c, out_c, mid_c, kernel_size=5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.hidden_c = mid_c
        self.conv1 = ConvBNAct(in_c, mid_c, 1, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.mp2 = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.mp3 = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = ConvBNAct(4 * mid_c, out_c, 1, 1)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(m(y[-1]) for m in [self.mp1, self.mp2, self.mp3])
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return "SPPELAN", f"[{self.in_c}, {self.out_c}, {self.hidden_c}, {self.kernel_size}]"

class SPPFMLP(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size=kernel_size
        hidden_dim = in_c // 2
        self.conv1 = ConvBNAct(in_c, hidden_dim, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_dim, 1, 1)
        self.conv3 = ConvBNAct(hidden_dim * 5, out_c, 1, 1)
        self.mlp = nn.ModuleList(ChannelMixer(hidden_dim, 2.0) for _ in range(3))
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        for i, mlp in enumerate(self.mlp):
            y[i + 2] = mlp(y[i + 2])
        return self.conv3(torch.cat(y, 1))
    
    def get_module_info(self):
        return "SPPFMLP", f"[{self.in_c}, {self.out_c}, {self.kernel_size}]"

class SPPFRepViT(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size=kernel_size
        hidden_dim = in_c // 2
        self.hidden_c = hidden_dim
        self.conv1 = ConvBNAct(in_c, 2 * hidden_dim, 1, 1)
        self.conv2 = ConvBNAct(hidden_dim * 5, out_c, 1, 1)
        self.vajra_rep_vit = nn.ModuleList(VajraRepViTBlock(hidden_dim, hidden_dim, False, 1, True, 5, 1.2, False) for _ in range(3))
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [a, b]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        for i, rep_vit in enumerate(self.vajra_rep_vit):
            y[i + 2] = rep_vit(y[i + 2])
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        for i, rep_vit in enumerate(self.vajra_rep_vit):
            y[i + 2] = rep_vit(y[i + 2])
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "SPPFRepViT", f"[{self.in_c}, {self.out_c}, {self.kernel_size}]"

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upsample(x)
    
    def get_module_info(self):
        return "Upsample", f"[{self.scale_factor}, {self.mode}]"

class SanlayanSPPF(nn.Module):
    def __init__(self, in_c, out_c, stride=2, expansion_ratio=1.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        total_c = sum(in_c)
        self.expansion_ratio = expansion_ratio
        self.out_channels = int(out_c * expansion_ratio)
        self.sppf = SPPF(in_c=total_c, out_c=self.out_channels, kernel_size=5)
    
    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        out = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs]
        concatenated_out = torch.cat(out, dim=1)
        sppf = self.sppf(concatenated_out)
        return sppf

    def get_module_info(self):
        return f"SanlayanSPPF", f"[{self.in_c}, {self.out_c}, {self.stride}]"
    
class SanlayanSPPFAttention(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.attn = nn.ModuleList(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, mlp_ratio=1.5) for _ in range(num_blocks)) #nn.ModuleList(
                #nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=2.0) for _ in range(2))) for _ in range(num_blocks)
            #)
        
        self.conv = ConvBNAct(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        fm2 = fm2 + fm1
        sppf = self.sppf(fm2)
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttention", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"

class VajraV1Attention(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, lite=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.conv1 = ConvBNAct(self.in_c, 3 * self.hidden_c, 1, 1)
        self.attn_branch1 = nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=1.2) for _ in range(2 * num_blocks)))
        self.attn_branch2 = nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=1.2) for _ in range(2 * num_blocks)))
        self.conv2 = ConvBNAct(3 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        branch1, branch2, branch3 = self.conv1(x).chunk(3, 1)
        attn_branch1 = self.attn_branch1(branch2)
        branch3 = attn_branch1 + branch3
        attn_branch2 = self.attn_branch2(branch3)
        return self.conv2(torch.cat((branch1, attn_branch1, attn_branch2), 1))
    
    def forward_split(self, x):
        branch1, branch2, branch3 = self.conv1(x).split(self.hidden_c, 1)
        attn_branch1 = self.attn_branch1(branch2)
        branch3 = attn_branch1 + branch3
        attn_branch2 = self.attn_branch2(branch3)
        return self.conv2(torch.cat((branch1, attn_branch1, attn_branch2), 1))

    def get_module_info(self):
        return f"VajraV1Attention", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.lite}]"

class ResidualBlock(nn.Module):
    def __init__(self, module, channels):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(0.01 * torch.ones(channels), requires_grad=True)

    def forward(self, x):
        return x + self.gamma.view(-1, len(self.gamma), 1, 1) * self.module(x)

class VajraV1AttentionBhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, area=4, mlp_ratio=1.5, lx=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.area = area
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.conv1 = ConvBNAct(self.in_c, 2 * self.hidden_c, 1, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(out_c), requires_grad=True) if lx else None
        self.attn = nn.ModuleList(
            nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads = self.hidden_c // 32 if not lite else self.hidden_c // 8, kernel_size=5, mlp_ratio=mlp_ratio, area=area) for _ in range(2))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        y = [fm1, fm2]
        y.extend(attn(y[-1]) for attn in self.attn)
        #fm = torch.cat((fm1, fm2), 1)
        #fm = torch.sum(torch.stack([fm, x]), dim=0)
        #y.append(fm)
        y = self.conv2(torch.cat(y, 1))
        return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y if self.gamma is not None else y
    
    def forward_split(self, x):
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        y = [fm1, fm2]
        y.extend(attn(y[-1]) for attn in self.attn)
        #fm = torch.cat((fm1, fm2), 1)
        #fm = torch.sum(torch.stack([fm, x]), dim=0)
        #y.append(fm)
        y = self.conv2(torch.cat(y, 1))
        return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y if self.gamma is not None else y

    def get_module_info(self):
        return f"VajraV1AttentionBhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}, {self.area}, {self.mlp_ratio}]"
    
class VajraV1AttentionBhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, area=4, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.area = area
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.conv1 = ConvBNAct(self.in_c, 2 * self.hidden_c, 1, 1)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads = self.hidden_c // 32 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio, area=area) for _ in range(4))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        y = [fm2]
        y.extend(attn(y[-1]) for attn in self.attn)
        fm = torch.cat((fm1, fm2), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        y = [fm2]
        y.extend(attn(y[-1]) for attn in self.attn)
        fm = torch.cat((fm1, fm2), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}, {self.area}, {self.mlp_ratio}]"
    
class VajraV1AttentionBhag3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=1.2, area=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.area = area
        self.conv1 = ConvBNAct(self.in_c, self.hidden_c, 1, 1)
        self.conv2 = ConvBNAct(self.in_c, self.hidden_c, 1, 1)
        self.attn =  nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio, area=area) for _ in range(num_blocks)))
        self.conv3 = ConvBNAct(3 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        c = self.attn(b)
        y = [a, b, c]
        return self.conv3(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag3", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}, {self.mlp_ratio}, {self.area}]"
    
class VajraV1AttentionBhag4(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, area=64, hd=32) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.area = area
        self.lite = lite
        self.sppf = SPPF(self.in_c, 2*self.hidden_c, 5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads = self.hidden_c // 16 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=1.2, area=area) for _ in range(4))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct(in_c + (num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        fm1, fm2 = self.sppf(x).chunk(2, 1)
        fm2 = fm2 + fm1
        fms = [x, fm1, fm2]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv2(torch.cat(fms, 1))
        return out
    
    def forward_split(self, x):
        fm1, fm2 = self.sppf(x).split(self.hidden_c, 1)
        fm2 = fm2 + fm1
        fms = [x, fm1, fm2]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv2(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"VajraV1AttentionBhag4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}, {self.area}]"

class VajraV1AttentionBhag5(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=2.0) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.sppf = SPPF(self.in_c, self.in_c, 5) #VajraSPPModule(self.in_c, self.in_c, 5)
        self.conv1 = ConvBNAct(self.in_c, 2 * self.hidden_c, 1, 1)
        self.attn = nn.Sequential(*(AttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, shortcut=True) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct(2 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))
    
    def forward_split(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag5", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}, {self.mlp_ratio}]"

class VajraV1AttentionBhag6(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=2.0, elan=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.sppf = SPPF(self.in_c, self.in_c, 5) if not elan else SPPELAN(self.in_c, self.in_c, self.hidden_c, 5) #VajraSPPModule(self.in_c, self.in_c, 5)
        self.conv1 = ConvBNAct(self.in_c, 2 * self.hidden_c, 1, 1)
        self.attn = nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=3, mlp_ratio=mlp_ratio) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct(2 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))
    
    def forward_split(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag6", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}]"

class VajraV1AttentionBhag7(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.conv1 = ConvBNAct(self.in_c, 2*self.hidden_c, 1, 1) #SPPF(self.in_c, 2*self.hidden_c, 5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=5, mlp_ratio=1.5) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        #fm2 = fm2 + fm1
        fms = [fm1, fm2]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv2(torch.cat(fms, 1))
        return out
    
    def forward_split(self, x):
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        #fm2 = fm2 + fm1
        fms = [fm1, fm2]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv2(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"VajraV1AttentionBhag7", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}]"
    
class VajraV1AttentionBhag8(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(self.in_c, 2*self.hidden_c, 5)
        self.attn = nn.ModuleList(
            nn.Sequential(*(AttentionBlockV3(self.hidden_c, self.hidden_c, num_heads = self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=5, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        y = list(self.sppf(x).chunk(2, 1))
        y.extend(attn(y[-1]) for attn in self.attn)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = self.sppf(x).split((self.hidden_c, self.hidden_c), 1)
        y = [y[0], y[1]]
        y.extend(attn(y[-1]) for attn in self.attn)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag8", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}]"

class VajraV1AttentionBhag9(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, mlp_ratio=1.5, area=4, expansion_ratio = 0.5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden_c = int(out_c * expansion_ratio)
        self.expansion_ratio = expansion_ratio
        self.mlp_ratio = mlp_ratio

        self.conv1 = ConvBNAct(in_c, 2 * self.hidden_c, 1, 1)
        self.attn_blocks = nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 32, kernel_size=9, mlp_ratio=mlp_ratio, area=area) for _ in range(2)))
        
        self.blocks = nn.ModuleList(InnerBlock(self.hidden_c, self.hidden_c, num_blocks=2, shortcut=True, kernel_size=1, expansion_ratio=0.5) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 3) * self.hidden_c, self.out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.blocks)
        y.append(self.attn_blocks(y[-1]))
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = self.conv1(x).split((self.hidden_c, self.hidden_c), 1)
        y = [y[0], y[1]]
        y.extend(block(y[-1]) for block in self.blocks)
        y.append(self.attn_blocks(y[-1]))
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV1AttentionBhag9", f"[{self.in_c}, {self.out_c}, {self.expansion_ratio}, {self.mlp_ratio}]"

class VajraV1AttentionBhag10(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=1.5, area=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.conv1 = ConvBNAct(self.in_c, 2*self.hidden_c, 1, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(InnerBlock(self.hidden_c, self.hidden_c, num_blocks=2, shortcut=True, kernel_size=1, expansion_ratio=0.5), AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=9, mlp_ratio=mlp_ratio, area=area)) for _ in range(num_blocks)
        )
        #self.blocks2 = ResidualBlock(nn.Sequential(InnerBlock(self.hidden_c, self.hidden_c, num_blocks=2, shortcut=True, kernel_size=1, expansion_ratio=0.5), AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio, area=area)), self.hidden_c)
        self.conv2 = ConvBNAct(4 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(blocks(y[-1]) for blocks in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = self.conv1(x).split((self.hidden_c, self.hidden_c), 1)
        y = [y[0], y[1]]
        y.extend(blocks(y[-1]) for blocks in self.blocks)
        return self.conv2(torch.cat(y, 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag10", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}]"

class VajraV1AttentionBhag11(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.conv1 = SPPF(self.in_c, 3*self.hidden_c, 5)
        self.blocks1 = ResidualBlock(nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio) for _ in range(2))), self.hidden_c)
        self.blocks1_final = AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio)
        self.blocks2 = ResidualBlock(nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio) for _ in range(2))), self.hidden_c)
        self.blocks2_final = AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio)
        self.conv2 = ConvBNAct(7 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        branch1, branch2, branch3 = self.conv1(x).chunk(3, 1)
        branch2_blocks = self.blocks1(branch2)
        branch2_final = self.blocks1_final(branch2_blocks)
        branch3 = branch3 + branch2_final
        branch3_blocks = self.blocks2(branch3)
        branch3_final = self.blocks2_final(branch3_blocks)
        return self.conv2(torch.cat((branch1, branch2, branch3, branch2_blocks, branch2_final, branch3_blocks, branch3_final), 1))
    
    def forward_split(self, x):
        branch1, branch2, branch3 = self.conv1(x).split(self.hidden_c, 1)
        branch2_blocks = self.blocks1(branch2)
        branch2_final = self.blocks1_final(branch2_blocks)
        branch3 = branch3 + branch2_final
        branch3_blocks = self.blocks2(branch3)
        branch3_final = self.blocks2_final(branch3_blocks)
        return self.conv2(torch.cat((branch1, branch2, branch3, branch2_blocks, branch2_final, branch3_blocks, branch3_final), 1))

    def get_module_info(self):
        return f"VajraV1AttentionBhag11", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}]"
    
class VajraV1AttentionBhag12(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, mlp_ratio=1.5, area=4, expansion_ratio = 0.5):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden_c = int(out_c * expansion_ratio)
        self.expansion_ratio = expansion_ratio
        self.mlp_ratio = mlp_ratio
        self.num_blocks = num_blocks
        num_blocks_inner_block = num_blocks // 2

        self.conv1 = ConvBNAct(in_c, 2 * self.hidden_c, 1, 1)
        self.attn_blocks = nn.ModuleList(
            ResidualBlock(nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 32, kernel_size=9, mlp_ratio=mlp_ratio, area=area) for _ in range(2))), self.hidden_c) for _ in range(num_blocks)
        )
        
        self.blocks = nn.ModuleList(InnerBlock(self.hidden_c, self.hidden_c, num_blocks=2, kernel_size=1, expansion_ratio=0.5) for _ in range(num_blocks_inner_block))
        self.conv2 = ConvBNAct((num_blocks_inner_block + num_blocks + 2) * self.hidden_c, self.out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.blocks)
        y.extend(attn_blocks(y[-1]) for attn_blocks in self.attn_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = self.conv1(x).split((self.hidden_c, self.hidden_c), 1)
        y = [y[0], y[1]]
        y.extend(block(y[-1]) for block in self.blocks)
        y.extend(attn_blocks(y[-1]) for attn_blocks in self.attn_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV1AttentionBhag12", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.mlp_ratio}]"

class VajraV2AttentionBhag1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, lite=False, scaling=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.hidden_c = out_c // 2
        self.scaling = scaling
        init_value = 0.01
        self.gamma = init_value * torch.values((out_c), requires_grad = True) if scaling else None
        self.conv1 = ConvBNAct(self.in_c, self.hidden_c, 1, 1)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 32 if not lite else self.hidden_c // 8, kernel_size=7, shortcut=True, area=4) for _ in range(4))) for _ in range(num_blocks)
            )
        self.conv2 = ConvBNAct((num_blocks + 1) * self.hidden_c, out_c, 1, 1)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(attn(y[-1]) for attn in self.attn)
        if self.gamma is not None:
            return x + self.gamma(1, -1, 1, 1) * self.conv2(torch.cat(y, 1))
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return f"VajraV2AttentionBhag1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}]"
    
class VajraV2AttentionBhag2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        assert in_c == 2 * self.hidden_c, "in_c needs to be equal to 2 * hidden_c for fusion"
        self.sppf = SPPF(self.in_c, 2 * self.hidden_c, 5)
        self.attn = nn.ModuleList(nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        fm1, fm2 = self.sppf(x).chunk(2, 1)
        y = [fm2]
        y.extend(attn(y[-1]) for attn in self.attn)
        fm = torch.cat((fm1, fm2), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        fm1, fm2 = self.sppf(x).split(self.hidden_c, 1)
        y = [fm2]
        y.extend(attn(y[-1]) for attn in self.attn)
        fm = torch.cat((fm1, fm2), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))

    def get_module_info(self):
        return f"VajraV2AttentionBhag2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}, {self.mlp_ratio}]"

class SanlayanSPPFAttentionBhag2(nn.Module): 
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(self.branch_a_channels, self.hidden_c)
        self.attn1 = nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=3, shortcut=True, area=1) for _ in range(num_blocks)))
        self.inner_block_modules = nn.ModuleList(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, shortcut=True, area=1) for _ in range(num_blocks))
        self.conv = ConvBNAct((num_blocks + 3) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        #downsample = inputs[0][:, :, ::self.stride, ::self.stride]
        #concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2, fm3 = concatenated_in.split((self.branch_a_channels, self.hidden_c, self.hidden_c), 1)
        sppf = self.sppf(fm1)
        fm2 = fm2 + sppf
        attn1 = self.attn1(fm2)
        fm3 = fm3 + attn1
        fms = [sppf, attn1, fm3]
        fms.extend(inner_block_attn(fms[-1]) for inner_block_attn in self.inner_block_modules)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionBhag2", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}, {self.lite}]"

class SanlayanSPPFAttentionV2(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, use_sppf=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.use_sppf = use_sppf
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5) if use_sppf else nn.Identity()
        self.attn = nn.ModuleList(AttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8) for _ in range(num_blocks))
        self.conv = ConvBNAct(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs]
        #out = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs]
        concatenated_in = torch.cat(downsample, dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        sppf = self.sppf(fm2) if self.use_sppf else fm2
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionV2", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"

class SanlayanSPPFAttentionV3(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, use_sppf=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = in_c // 2
        self.out_c = out_c
        self.lite = lite
        self.use_sppf = use_sppf
        self.sppf = SPPF(in_c=self.in_c, out_c=self.in_c, kernel_size=5) if use_sppf else nn.Identity()
        self.attn = nn.ModuleList(AttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8) for _ in range(num_blocks))
        self.conv = ConvBNAct(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs] if self.stride > 1 else [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs[:-1]] + inputs[-1:]
        concatenated_in = torch.sum(torch.stack(downsample), dim=0) #torch.cat(downsample, dim=1)
        #fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        sppf = self.sppf(concatenated_in) if self.use_sppf else concatenated_in
        fm3, fm4 = sppf.chunk(2, 1)
        fm4 = fm4 + fm3
        fms = [fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionV3", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"
    
class SanlayanSPPFAttentionV4(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=2.0) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv = ConvBNAct(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        sppf = self.sppf(fm2)
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionV4", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"

class SanlayanSPPFAttentionV5(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=5, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv = ConvBNAct(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        if self.branch_a_channels == self.out_c:
            fm2 = fm2 + fm1
        sppf = self.sppf(fm2)
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionV5", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"

class SanlayanSPPFAttentionV6(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPFMakkarNorm(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AreaAttentionBlockV3(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=5, mlp_ratio=mlp_ratio, area=1) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv = ConvMakkarNorm(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        if self.branch_a_channels == self.out_c:
            fm2 = fm2 + fm1
        sppf = self.sppf(fm2)
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionV6", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"

class SanlayanSPPFAttentionV7(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.attn = nn.ModuleList(
                nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
            )
        self.conv = ConvBNAct(in_c + (num_blocks) * self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = F.interpolate(inputs[0], size=(H, W), mode="nearest")
        concatenated_in = torch.cat((downsample, inputs[-1]), dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        if self.branch_a_channels == self.out_c:
            fm2 = fm2 + fm1
        sppf = self.sppf(fm2)
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4]
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFAttentionV7", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}]"

class SanlayanSPPFVajraMerudanda(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, use_sppf=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = out_c // 2
        self.out_c = out_c
        self.lite = lite
        self.use_sppf = use_sppf
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5) if use_sppf else nn.Identity()
        self.vajra_merudanda = VajraV2MerudandaBhag3(self.hidden_c, self.hidden_c, num_blocks, True, 0.5, True) #nn.ModuleList(InnerBlock(self.hidden_c, self.hidden_c, 2, True, 1, 0.25, False) for _ in range(num_blocks)) #VajraV1MerudandaBhag2(self.hidden_c, self.hidden_c, num_blocks, 1, True, 0.5, inner_block=True)
        self.conv = ConvBNAct(in_c + self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs]
        concatenated_in = torch.cat(downsample, dim=1)
        fm1, fm2 = concatenated_in.split((self.branch_a_channels, self.out_c), 1)
        sppf = self.sppf(fm2) if self.use_sppf else fm2
        fm3, fm4 = sppf.split(self.hidden_c, 1)
        fm4 = fm4 + fm3
        fms = [fm1, fm3, fm4, self.vajra_merudanda(fm4)]
        #fms.extend(inner_block(fms[-1]) for inner_block in self.inner_block)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFVajraMerudanda", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}, {self.lite}, {self.use_sppf}]"

class SanlayanSPPFVajraMerudandaV2(nn.Module):
    def __init__(self, in_c, out_c, stride=2, num_blocks=2, lite=False, use_sppf=True, inner_block=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = in_c // 2
        self.out_c = out_c
        self.lite = lite
        self.use_sppf = use_sppf
        self.inner_block = inner_block
        self.sppf = SPPF(in_c=self.in_c, out_c=self.in_c, kernel_size=5) if use_sppf else nn.Identity()
        self.vajra_merudanda = VajraV2MerudandaBhag3(self.hidden_c, self.hidden_c, num_blocks, True, 0.5, inner_block=inner_block)
        #self.attn = nn.ModuleList(AttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8) for _ in range(num_blocks))
        self.conv = ConvBNAct(in_c + self.hidden_c, out_c, 1, 1)

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        downsample = [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs] if self.stride > 1 else [F.interpolate(inp, size=(H, W), mode="nearest") for inp in inputs[:-1]] + inputs[-1:]
        concatenated_in = torch.sum(torch.stack(downsample), dim=0)
        sppf = self.sppf(concatenated_in) if self.use_sppf else concatenated_in
        fm3, fm4 = sppf.chunk(2, 1)
        fm4 = fm4 + fm3
        fms = [fm3, fm4, self.vajra_merudanda(fm4)]
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFVajraMerudandaV2", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}, {self.inner_block}]"

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

        self.mixconv = MixConv2d(dim,dim,k=dwconv_kernels) #BiDWConv(dim, dim, dwconv_kernels)#nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
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
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = ConvBNAct(dim, h, kernel_size=1, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None)
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=3, groups=dim, act=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2) #qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Attention scores
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.positional_encoding(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    
USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
        LOGGER.info(f"Flash Attention available!\n")
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        LOGGER.info(f"Flash Attention not available on this device\n")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    LOGGER.info(f"Flash Attention not available on this device\n")

class AttentionV2(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * 0.5)
        self.scale = self.head_dim ** -0.5

        self.qk = ConvBNAct(dim, 2 * dim, kernel_size=1, act=None)
        self.v = ConvBNAct(dim, dim, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None)
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, groups=dim, act=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qk = self.qk(x).view(B, self.num_heads, self.head_dim * 2, N)
        v = self.v(x)
        positional_encoding = self.positional_encoding(v)
        v = v.view(B, self.num_heads, self.head_dim, N)
        q, k = qk.split([self.head_dim, self.head_dim], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.permute(0, 1, 3, 2)
            k = k.permute(0, 1, 3, 2)
            v = v.permute(0, 1, 3, 2)
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W)

        return self.proj(x + positional_encoding)

class AttentionV3(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = ConvBNAct(dim, 3 * dim, kernel_size=1, act=None)
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0) if USE_FLASH_ATTN else nn.Identity()
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0) if USE_FLASH_ATTN else nn.Identity()
        self.proj = RepConv(dim, dim, stride=1, kernel_size=3, act=None)
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, groups=dim, act=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, self.num_heads, self.head_dim * 3, N)
        q, k, v = qkv.split([self.head_dim, self.head_dim, self.head_dim], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.permute(0, 1, 3, 2)
            k = k.permute(0, 1, 3, 2)
            v = v.permute(0, 1, 3, 2)
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = self.talking_head1(attn)
            attn = F.softmax(attn, dim=-1)
            attn = self.talking_head2(attn)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W)

        positional_encoding = self.positional_encoding(v.reshape(B, C, H, W))

        return self.proj(x + positional_encoding)

class AreaAttention(nn.Module):

    def __init__(self, dim, num_heads=8, kernel_size=3, area=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.area = area
        self.scale = self.head_dim ** -0.5
        self.qk = ConvBNAct(dim, 2 * dim, kernel_size=1, act=None)
        self.v = ConvBNAct(dim, dim, 1, 1, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None)
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, act=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        #qkv = self.qkv(x).flatten(2).transpose(1, 2)

        #if self.area > 1:
            #qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            #B, N, _ = qkv.shape
        
        #q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split([self.head_dim, self.head_dim, self.head_dim], dim=-1)

        if x.is_cuda and USE_FLASH_ATTN:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C*2)
                v = v.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = qk.split([C, C], dim=2)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            out = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)

            if self.area > 1:
                out = out.reshape(B // self.area, N * self.area, C)
                B, N, _ = out.shape
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        #elif x.is_cuda and not USE_FLASH_ATTN:
            #out = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=None, dropout_p=0.0, is_causal=False)
            #out = out.permute(0, 2, 1, 3)
        else:
            qk = self.qk(x).flatten(2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, C*2, N // self.area)
                v = v.reshape(B * self.area, C, N // self.area)
                B, _, N = qk.shape
            
            q, k = qk.split([C, C], dim=1)
            q = q.view(B, self.num_heads, self.head_dim, N)
            k = k.view(B, self.num_heads, self.head_dim, N)
            v = v.view(B, self.num_heads, self.head_dim, N)
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            #max_attn = attn.max(dim=-1, keepdim=True).values
            #exp_attn = torch.exp(attn - max_attn)
            #attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            out = (v @ attn.transpose(-2, -1))

            if self.area > 1:
                out = out.reshape(B // self.area, C, N * self.area)
                B, _, N = out.shape
            out = out.reshape(B, C, H, W)
            #q = q.permute(0, 2, 3, 1)
            #k = k.permute(0, 2, 3, 1)
            #v = v.permute(0, 2, 3, 1)
            #attn = (q.transpose(-2, -1) @ k) * self.scale
            #attn = F.softmax(attn, dim=-1)
            #out = (v @ attn.transpose(-2, -1))
            #out = out.permute(0, 3, 1, 2) # (B, N, self.num_heads, self.head_dim)
            #v = v.permute(0, 3, 1, 2)

        #if self.area > 1:
            #out = out.reshape(B // self.area, N * self.area, C)
            #v = v.reshape(B // self.area, N * self.area, C)
            #B, N, _ = out.shape
        
        #out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        #v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        #out = out + self.positional_encoding(v)
        #out = self.proj(out)
        return self.proj(out + positional_encoding)

class AreaAttentionV2(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3, area=4, local_branch=True, dyt=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.area = area
        self.scale = self.head_dim ** -0.5
        self.qk = ConvBNAct(dim, 2 * dim, kernel_size=1, act=None) if not dyt else ConvDyTAct(dim, 2 * dim, kernel_size=1, act=None)
        self.v = ConvBNAct(dim, dim, 1, 1, act=None) if not dyt else ConvDyTAct(dim, dim, 1, 1, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None) if not dyt else ConvDyTAct(dim, dim, 1, 1, act=None)
        self.local_branch = local_branch
        self.conv_branch = ConvBNAct(dim, dim, stride=1, kernel_size=3, groups=dim) if local_branch else nn.Identity()
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, act=None) if not dyt else ConvDyTAct(dim, dim, 1, kernel_size, groups=dim, act=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        if self.local_branch:
            conv = self.conv_branch(x)

        if x.is_cuda and USE_FLASH_ATTN:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C*2)
                v = v.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = qk.split([C, C], dim=2)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            out = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)

            if self.area > 1:
                out = out.reshape(B // self.area, N * self.area, C)
                B, N, _ = out.shape
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        else:
            qk = self.qk(x).flatten(2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, C*2, N // self.area)
                v = v.reshape(B * self.area, C, N // self.area)
                B, _, N = qk.shape
            
            q, k = qk.split([C, C], dim=1)
            q = q.view(B, self.num_heads, self.head_dim, N)
            k = k.view(B, self.num_heads, self.head_dim, N)
            v = v.view(B, self.num_heads, self.head_dim, N)
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (v @ attn.transpose(-2, -1))

            if self.area > 1:
                out = out.reshape(B // self.area, C, N * self.area)
                B, _, N = out.shape
            out = out.reshape(B, C, H, W)
            
        out = self.proj(out + positional_encoding)

        if self.local_branch:
            out = out + conv
        
        return out
    
class AreaAttentionV3(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3, area=4, local_branch=True, makkar_norm=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.area = area
        self.scale = self.head_dim ** -0.5
        self.qk = ConvBNAct(dim, 2 * dim, kernel_size=1, act=None) if not makkar_norm else ConvMakkarNorm(dim, 2 * dim, kernel_size=1)
        self.v = ConvBNAct(dim, dim, 1, 1, act=None) if not makkar_norm else ConvDyTAct(dim, dim, 1, 1, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None) if not makkar_norm else ConvMakkarNorm(dim, dim, 1, 1)
        self.local_branch = local_branch
        self.conv_branch = ConvMakkarNorm(dim, dim, 1, 3, groups=4) if local_branch else nn.Identity()
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, act=None) if not makkar_norm else ConvMakkarNorm(dim, dim, 1, kernel_size, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        if self.local_branch:
            conv = self.conv_branch(x)

        if x.is_cuda and USE_FLASH_ATTN:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C*2)
                v = v.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = qk.split([C, C], dim=2)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            out = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)

            if self.area > 1:
                out = out.reshape(B // self.area, N * self.area, C)
                B, N, _ = x.shape
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        else:
            qk = self.qk(x).flatten(2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, C*2, N // self.area)
                v = v.reshape(B * self.area, C, N // self.area)
                B, _, N = qk.shape
            
            q, k = qk.split([C, C], dim=1)
            q = q.view(B, self.num_heads, self.head_dim, N)
            k = k.view(B, self.num_heads, self.head_dim, N)
            v = v.view(B, self.num_heads, self.head_dim, N)
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (v @ attn.transpose(-2, -1))

            if self.area > 1:
                out = out.reshape(B // self.area, C, N * self.area)
                B, _, N = out.shape
            out = out.reshape(B, C, H, W)
            
        out = self.proj(out + positional_encoding)

        if self.local_branch:
            out = out + conv
        
        return out

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = ConvBNAct(dim, all_head_dim * 2, 1, 1, act=None)
        self.v = ConvBNAct(dim, all_head_dim, 1, 1, act=None)
        self.proj = ConvBNAct(all_head_dim, dim, 1, 1, act=None)

        self.pe = ConvBNAct(all_head_dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, act=None)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        if x.is_cuda and USE_FLASH_ATTN:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)
            pp = self.pe(v)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C * 2)
                v = v.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = qk.split([C, C], dim=2)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)

            if self.area > 1:
                x = x.reshape(B // self.area, N * self.area, C)
                B, N, _ = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            qk = self.qk(x).flatten(2)
            v = self.v(x)
            pp = self.pe(v)
            v = v.flatten(2)
            if self.area > 1:
                qk = qk.reshape(B * self.area, C * 2, N // self.area)
                v = v.reshape(B * self.area, C, N // self.area)
                B, _, N = qk.shape

            q, k = qk.split([C, C], dim=1)
            q = q.view(B, self.num_heads, self.head_dim, N)
            k = k.view(B, self.num_heads, self.head_dim, N)
            v = v.view(B, self.num_heads, self.head_dim, N)
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            if self.area > 1:
                x = x.reshape(B // self.area, C, N * self.area)
                B, _, N = x.shape
            x = x.reshape(B, C, H, W)

        return self.proj(x + pp)
    
class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(ConvBNAct(dim, mlp_hidden_dim, 1, 1), ConvBNAct(mlp_hidden_dim, dim, 1, 1, act=None))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    
class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32
        self.c1 = c1
        self.c2 = c2
        self.num_blocks = n
        self.a2_bool = a2
        self.area = area
        self.residual_bool = residual
        self.mlp_ratio=mlp_ratio

        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct((1 + n) * c_, c2, 1, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "A2C2f", f"[{self.c1}, {self.c2}, {self.num_blocks}, {self.a2_bool}, {self.area}, {self.residual_bool}, {self.mlp_ratio}]"

class AxialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = ConvBNAct(dim, 3 * dim, kernel_size=1, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None)
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, groups=dim, act=None)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).view(B, self.num_heads, 3 * self.head_dim, H, W)
        q, k, v = qkv.split([self.head_dim, self.head_dim, self.head_dim], dim=2)

        q_h = q.permute(0, 1, 3, 4, 2)
        k_h = k.permute(0, 1, 3, 4, 2)
        v_h = v.permute(0, 1, 3, 4, 2)

        q_h = q_h.reshape(B * self.num_heads * W, H, self.head_dim)
        k_h = k_h.reshape(B * self.num_heads * W, H, self.head_dim)
        v_h = v_h.reshape(B * self.num_heads * W, H, self.head_dim)

        attn_h = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        attn_h = F.softmax(attn_h, dim=-1)
        out_h = torch.matmul(attn_h, v_h) # (B * self.num_heads * W, H, self.head_dim)
        out_h = out_h.reshape(B, self.num_heads, W, H, self.head_dim).permute(0, 1, 4, 3, 2) # Reshape back to (B, self.num_heads, W, H, self.head_dim) then to (B, self.num_heads, self.head_dim, H, W)

        q_w = q.permute(0, 1, 3, 4, 2)
        k_w = k.permute(0, 1, 3, 4, 2)
        v_w = v.permute(0, 1, 3, 4, 2)

        q_w = q_w.permute(0, 1, 3, 2, 4).reshape(B * self.num_heads * H, W, self.head_dim)
        k_w = k_w.permute(0, 1, 3, 2, 4).reshape(B * self.num_heads * H, W, self.head_dim)
        v_w = v_w.permute(0, 1, 3, 2, 4).reshape(B * self.num_heads * H, W, self.head_dim)

        attn_w = torch.matmul(q_w, k_w.transpose(-2, -1)) * self.scale
        attn_w = F.softmax(attn_w, dim=-1) # (B * self.num_heads * W, H, self.head_dim)
        out_w = torch.matmul(attn_w, v_w)
        out_w = out_w.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 4, 2, 3) # (B, self.num_heads, self.head_dim, H, W)

        out = out_h + out_w
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        v_combined = v.reshape(B, self.num_heads * self.head_dim, H, W)
        out = out + self.positional_encoding(v_combined)
        out = self.proj(out)

        return out

class AxialAttentionV2(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=7, local_branch=True):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.local_branch = local_branch
        self.qk = ConvBNAct(dim, 2 * dim, kernel_size=1, act=None)
        self.v = ConvBNAct(dim, dim, 1, 1, act=None)
        self.proj = ConvBNAct(dim, dim, kernel_size=1, act=None)
        self.wh = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.ww = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.positional_encoding = ConvBNAct(dim, dim, kernel_size=kernel_size, groups=dim, act=None)
        self.cross_conv = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1, bias=False)

        self.conv_branch = ConvBNAct(dim, dim, 1, 3, groups=4) if local_branch else nn.Identity()
        self.pw_conv = ConvBNAct(2 * dim, dim, 1, 1) if local_branch else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if not hasattr(self, 'pos_emb_h'):
            self.pos_emb_h = nn.Parameter(torch.randn(1, 1, self.head_dim, H, 1))
            self.pos_emb_w = nn.Parameter(torch.randn(1, 1, self.head_dim, 1, W))

        if self.local_branch:
            conv = self.conv_branch(x)
        
        qk = self.qk(x).view(B, self.num_heads, 2 * self.head_dim, H, W)
        q, k = qk.split([self.head_dim, self.head_dim], dim=2)
        v = self.v(x)
        positional_encoding = self.positional_encoding(v)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W)

        q = q + self.pos_emb_h + self.pos_emb_w
        k = k + self.pos_emb_h + self.pos_emb_w
        v = v + self.pos_emb_h + self.pos_emb_w

        q_h = q.permute(0, 1, 3, 4, 2)
        k_h = k.permute(0, 1, 3, 4, 2)
        v_h = v.permute(0, 1, 3, 4, 2)

        q_h = q_h.reshape(B * self.num_heads * W, H, self.head_dim)
        k_h = k_h.reshape(B * self.num_heads * W, H, self.head_dim)
        v_h = v_h.reshape(B * self.num_heads * W, H, self.head_dim)

        attn_h = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        attn_h = F.softmax(attn_h, dim=-1)
        out_h = torch.matmul(attn_h, v_h) # (B * self.num_heads * W, H, self.head_dim)
        out_h = out_h.reshape(B, self.num_heads, W, H, self.head_dim).permute(0, 1, 4, 3, 2) # Reshape back to (B, self.num_heads, W, H, self.head_dim) then to (B, self.num_heads, self.head_dim, H, W)

        out_h_transform = self.cross_conv(out_h.reshape(B * self.num_heads, self.head_dim, H, W))
        out_h_transform = out_h_transform.view(B, self.num_heads, self.head_dim, H, W)
        q_w = q + out_h_transform

        q_w = q.permute(0, 1, 3, 4, 2)
        k_w = k.permute(0, 1, 3, 4, 2)
        v_w = v.permute(0, 1, 3, 4, 2)

        q_w = q_w.permute(0, 1, 3, 2, 4).reshape(B * self.num_heads * H, W, self.head_dim)
        k_w = k_w.permute(0, 1, 3, 2, 4).reshape(B * self.num_heads * H, W, self.head_dim)
        v_w = v_w.permute(0, 1, 3, 2, 4).reshape(B * self.num_heads * H, W, self.head_dim)

        attn_w = torch.matmul(q_w, k_w.transpose(-2, -1)) * self.scale
        attn_w = F.softmax(attn_w, dim=-1) # (B * self.num_heads * W, H, self.head_dim)
        out_w = torch.matmul(attn_w, v_w)
        out_w = out_w.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 4, 2, 3) # (B, self.num_heads, self.head_dim, H, W)

        out = self.wh * out_h + self.ww * out_w
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        #v_combined = v.reshape(B, self.num_heads * self.head_dim, H, W)
        out = out + positional_encoding
        out = self.proj(out)
        if self.local_branch:
            out = self.pw_conv(torch.cat([out, conv], 1))

        return out

class AreaAxialAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, kernel_size=3, grid_size = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.grid_h = self.grid_w = int(grid_size ** 0.5)
        self.axial_attn = AxialAttention(dim, num_heads, kernel_size)

    def forward(self, x):
        B, C, H, W = x.shape
        block_h = H // self.grid_h
        block_w = W // self.grid_w

        x_area = x.reshape(B, C, self.grid_h, block_h, self.grid_w, block_w)
        x_area = x_area.permute(0, 2, 4, 1, 3, 5).reshape(B * self.grid_size, C, block_h, block_w)

        out_area = self.axial_attn(x_area)
        out_area = out_area.reshape(B, self.grid_h, self.grid_w, C, block_h, block_w)
        out_area = out_area.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return out_area
    
class AttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, shortcut=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = Attention(in_c, num_heads=num_heads)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, 2 * in_c, 1, 1), ConvBNAct(2 * in_c, out_c, 1, act=None))
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AttentionBlock", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"
    
class AttentionBlockV2(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True, mlp_ratio=2., swiglu_ffn=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AttentionV2(in_c, num_heads=num_heads, kernel_size=kernel_size)
        mlp_hidden_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, mlp_hidden_dim, 1, 1), ConvBNAct(mlp_hidden_dim, out_c, 1, act=None)) if not swiglu_ffn else SwiGLUFFN(in_c, out_c, mlp_ratio)
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AttentionBlockV2", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"
    
class AttentionBlockV3(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=3, shortcut=True, mlp_ratio=2.) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AttentionV3(in_c, num_heads=num_heads, kernel_size=kernel_size)
        mlp_hidden_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, mlp_hidden_dim, 1, 1), ConvBNAct(mlp_hidden_dim, out_c, 1, act=None))
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AttentionBlockV2", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"

class AreaAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True, area=4, mlp_ratio=1.2, swiglu_ffn=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AreaAttention(in_c, num_heads=num_heads, kernel_size=kernel_size, area=area)
        mlp_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, mlp_dim, 1, 1), ConvBNAct(mlp_dim, out_c, 1, act=None)) if not swiglu_ffn else SwiGLUFFN(in_c, out_c, mlp_ratio)
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AreaAttentionBlock", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"

class AreaAttentionBlockV2(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True, area=4, mlp_ratio=1.2) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AreaAttentionV2(in_c, num_heads=num_heads, kernel_size=kernel_size, area=area)
        mlp_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, mlp_dim, 1, 1), ConvBNAct(mlp_dim, out_c, 1, act=None))
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AreaAttentionBlockV2", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"

class AreaAttentionBlockV3(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True, area=1, mlp_ratio=1.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AreaAttentionV3(in_c, num_heads=num_heads, kernel_size=kernel_size, area=area)
        mlp_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(ConvMakkarNorm(in_c, mlp_dim, 1, 1), ConvMakkarNorm(mlp_dim, out_c, 1))
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AreaAttentionBlockV3", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"

class AxialAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AxialAttention(in_c, num_heads=num_heads, kernel_size=kernel_size)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, 2 * in_c, 1, 1), ConvBNAct(2 * in_c, out_c, 1, act=None))
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AxialAttentionBlock", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"
    
class AreaAxialAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, grid_size=4, shortcut=True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AreaAxialAttention(in_c, num_heads=num_heads, kernel_size=kernel_size, grid_size=grid_size)
        self.conv2 = nn.Sequential(ConvBNAct(in_c, 2 * in_c, 1, 1), ConvBNAct(2 * in_c, out_c, 1, act=None))
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    def get_module_info(self):
        return f"AreaAxialAttentionBlock", f"[{self.in_c}, {self.out_c}, {self.num_heads}, {self.shortcut}]"
    
class AttentionBottleneck(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        hidden_c = int(out_c * 0.5)
        self.num_blocks = num_blocks
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size=kernel_size)
        self.attn = nn.ModuleList(AttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 64) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c + (num_blocks + 1) * hidden_c, out_c, 1, 1)

    def forward(self, x):
        y = [x, self.conv1(x)]
        y.extend(attn_block(y[-1]) for attn_block in self.attn)
        out = self.conv2(torch.cat(y, 1))
        return out
    
    def get_module_info(self):
        return f"AttentionBottleneck", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.kernel_size}]"

class AttentionBottleneckV2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        hidden_c = int(out_c * 0.5)
        self.num_blocks = num_blocks
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c // 2, hidden_c, 1, kernel_size=kernel_size)
        self.attn = nn.ModuleList(AttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 64) for _ in range(num_blocks))
        self.conv2 = ConvBNAct(in_c // 2 + (num_blocks + 1) * hidden_c, out_c, 1, 1)

    def forward(self, x):
        a, b = x.chunk(2, 1)
        y = [a, self.conv1(b)]
        y.extend(attn_block(y[-1]) for attn_block in self.attn)
        out = self.conv2(torch.cat(y, 1))
        return out
    
    def forward_split(self, x):
        a, b = x.split((self.in_c // 2, self.in_c // 2), 1)
        y = [a, self.conv1(b)]
        y.extend(attn_block(y[-1]) for attn_block in self.attn)
        out = self.conv2(torch.cat(y, 1))
        return out
    
    def get_module_info(self):
        return f"AttentionBottleneckV2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.kernel_size}]"

class AttentionBottleneckV3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        hidden_c = int(out_c * 0.5)
        self.num_blocks = num_blocks
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size=kernel_size)
        self.attn = nn.ModuleList(AttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 64) for _ in range(num_blocks))
        self.conv2 = ConvBNAct((num_blocks + 1) * hidden_c, out_c, 1, 1)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(attn_block(y[-1]) for attn_block in self.attn)
        out = self.conv2(torch.cat(y, 1))
        return out
    
    def get_module_info(self):
        return f"AttentionBottleneckV3", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.kernel_size}]"

class AttentionBottleneckV4(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, kernel_size=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        hidden_c = int(out_c * 0.5)
        assert in_c == out_c, "in channels and out channels must be equal for AttentionBottleneckV4"
        self.num_blocks = num_blocks
        self.kernel_size=kernel_size
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, kernel_size=kernel_size)
        self.attn = nn.Sequential(*(AttentionBlock(hidden_c, hidden_c, num_heads=hidden_c // 64) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct(2 * hidden_c, out_c, 1, 1)

    def forward(self, x):
        #y = [self.conv1(x)]
        a = self.conv1(x)
        b = self.attn(a)
        #y.extend(attn_block(y[-1]) for attn_block in self.attn)
        out = self.conv2(torch.cat((a, b), 1))
        return x + out
    
    def get_module_info(self):
        return f"AttentionBottleneckV4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.kernel_size}]"

class VajraV2Block(nn.Module):
    def __init__(self, dim, expansion_ratio=0.5, rep_vgg_k=5) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_c = int(expansion_ratio * dim)
        self.conv1 = nn.Sequential(DepthwiseConvBNAct(dim, dim, 1, 3), ConvBNAct(dim, self.hidden_c, 1, 1))
        self.conv2 = RepVGGDW(self.hidden_c, rep_vgg_k, 1) if rep_vgg_k > 3 else ConvBNAct(self.hidden_c, self.hidden_c, 1, 3) 
        self.conv3 = nn.Sequential(ConvBNAct(self.hidden_c, dim, 1, 1), DepthwiseConvBNAct(dim, dim, 1, 3))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return x + conv3
    
class VajraV1LiteBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=0.5, kernel_size=3) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_c = int(expansion_ratio * dim)
        self.dwconv1 = DWPyConv2(dim, self.hidden_c, [3, 3], True) #nn.Sequential(DepthwiseConvBNAct(self.dim, self.dim, 1, kernel_size=kernel_size), ConvBNAct(self.dim, self.hidden_c, 1, 1))
        self.dwconv2 = nn.Sequential(DepthwiseConvBNAct(self.hidden_c, self.hidden_c, 1, kernel_size=kernel_size), ConvBNAct(self.hidden_c, dim, 1, 1))
        self.conv2 = ConvBNAct(self.hidden_c + dim, dim, 1, 1)

    def forward(self, x):
        dwconv1 = self.dwconv1(x)
        dwconv2 = self.dwconv2(dwconv1)
        out = self.conv2(torch.cat((dwconv1, dwconv2), dim=1))
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

class MobileNetV3_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="silu", stride=1, pw=True):
        #input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        c_mid = e if e != None else c1
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.e = e
        self.sa = sa
        self.act = act
        self.stride=stride
        self.pw = pw
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else [] #if c_mid != c1 else []
        features.extend([mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                         #attn,
                         nn.Conv2d(c_mid, c2, 1),
                         nn.BatchNorm2d(c2),
                         #nn.SiLU(),
                         ])
        self.layers = nn.Sequential(*features)
    def forward(self, x):
        #print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)
        
    def get_module_info(self):
        return "LeYOLO_MobileNetV3_BLOCK", f"[{self.c1}, {self.c2}, {self.e}, {self.sa}, {self.act}, {self.stride}, {self.pw}]" 

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
    
class Concatenate(nn.Module):
    def __init__(self, in_c, dimension=1):
        super().__init__()
        self.in_c = in_c
        self.dim = dimension

    def forward(self, x):
        return torch.cat(x, self.dim)
    
    def get_module_info(self):
        return f"Concatenate", f"[{self.in_c}, {self.dim}]"

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
            attn_chs = make_divisible(in_c * radix * rd_ratio, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = ConvBNAct(in_c, mid_chs, stride=stride, kernel_size=kernel_size, groups=radix)
        self.fc1 = ConvBNAct(out_c, attn_chs, 1, 1, groups=radix)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
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
    def __init__(self, in_c, c_mid=256, out_c=32) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_c, c_mid, 1, 3)
        self.conv2 = ConvBNAct(c_mid, c_mid, 1, 3)
        self.conv3 = ConvBNAct(c_mid, out_c, 1, 1)
        self.upsample = nn.ConvTranspose2d(c_mid, c_mid, 2, 2, 0, bias=True)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))

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
    
class RepVGGBlock(nn.Module):
    def __init__(self, in_c, out_c, act="silu"):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = ConvBNAct(in_c, out_c, 1, 3, 1, act=None)
        self.conv2 = ConvBNAct(in_c, out_c, 1, 1, 0, act=None)
        self.act = nn.Identity() if act is None else act_table[act]

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)
    
    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.in_c, self.out_c, 3, 1, padding=1)
        
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])
        
    def _fuse_bn_tensor(self, branch: ConvBNAct):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t =  (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class RepCSP(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, expansion_ratio=1.0):
        super().__init__()
        self.block = RepVGGBlock
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(
            *[self.block(hidden_c, hidden_c, act="silu") for _ in range(num_blocks)]
        )
        self.conv3 = ConvBNAct(hidden_c, out_c, 1, 1) if hidden_c != out_c else nn.Identity()

    def forward(self, x):
        conv1 = self.conv1(x)
        blocks = self.bottleneck_blocks(conv1)
        conv2 = self.conv2(x)
        return self.conv3(blocks + conv2)

class CSP(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=3, expansion_ratio=1.0):
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.conv1 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.conv2 = ConvBNAct(in_c, hidden_c, 1, 1)
        self.bottleneck_blocks = nn.Sequential(
            *[ConvBNAct(hidden_c, hidden_c, 1, 3) for _ in range(num_blocks)]
        )
        self.conv3 = ConvBNAct(hidden_c, out_c, 1, 1) if hidden_c != out_c else nn.Identity()

    def forward(self, x):
        conv1 = self.conv1(x)
        blocks = self.bottleneck_blocks(conv1)
        conv2 = self.conv2(x)
        return self.conv3(blocks + conv2)

class RepNCSPELAN4(nn.Module):
    def __init__(self, in_c, out_c, mid_c1, mid_c2, num_blocks=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mid_c1 = mid_c1
        self.mid_c2 = mid_c2
        self.num_blocks = num_blocks
        self.hidden_c = mid_c1 // 2
        self.conv1 = ConvBNAct(in_c, mid_c1, 1, 1)
        self.conv2 = nn.Sequential(RepCSP(mid_c1 // 2, mid_c2, num_blocks), ConvBNAct(mid_c2, mid_c2, 1, 3))
        self.conv3 = nn.Sequential(RepCSP(mid_c2, mid_c2, num_blocks), ConvBNAct(mid_c2, mid_c2, 1, 3))
        self.conv4 = ConvBNAct(mid_c1 + (2 * mid_c2), out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(conv(y[-1]) for conv in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.conv1(x).split((self.hidden_c, self.hidden_c), 1))
        y.extend(conv(y[-1]) for conv in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))
    
    def get_module_info(self):
        return "RepNCSPELAN4", f"[{self.in_c}, {self.out_c}, {self.mid_c1}, {self.mid_c2}, {self.num_blocks}]"

class VajraV1MerudandaX(nn.Module):
    def __init__(self, in_c, out_c, mid_c1, mid_c2, num_blocks=1, rep_csp=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mid_c1 = mid_c1
        self.mid_c2 = mid_c2
        self.num_blocks = num_blocks
        self.hidden_c = mid_c1 // 2
        self.conv1 = ConvBNAct(in_c, mid_c1, 1, 1)
        self.conv2 = nn.Sequential(VajraV2InnerBlock(mid_c1//2, mid_c2, num_bottleneck_blocks=num_blocks, rep_bottleneck=False, shortcut=True), ConvBNAct(mid_c2, mid_c2, 1, 3)) if not rep_csp else nn.Sequential(RepCSP(mid_c1 // 2, mid_c2, num_blocks), ConvBNAct(mid_c2, mid_c2, 1, 3))
        self.conv3 = nn.Sequential(VajraV2InnerBlock(mid_c1//2, mid_c2, num_bottleneck_blocks=num_blocks, rep_bottleneck=False, shortcut=True), ConvBNAct(mid_c2, mid_c2, 1, 3)) if not rep_csp else nn.Sequential(RepCSP(mid_c2, mid_c2, num_blocks), ConvBNAct(mid_c2, mid_c2, 1, 3))
        self.conv4 = ConvBNAct(mid_c1 + (2 * mid_c2), out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(conv(y[-1]) for conv in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.conv1(x).split((self.hidden_c, self.hidden_c), 1))
        y.extend(conv(y[-1]) for conv in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV1MerudandaX", f"[{self.in_c}, {self.out_c}, {self.mid_c1}, {self.mid_c2}, {self.num_blocks}]"

class VajraV1AttentionX(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=2.0, elan=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.sppf = SPPF(self.in_c, self.out_c, 5) if not elan else SPPELAN(self.in_c, self.out_c, self.hidden_c, 5) #VajraSPPModule(self.in_c, self.in_c, 5)
        self.conv1 = ConvBNAct(self.out_c, 2 * self.hidden_c, 1, 1)
        self.attn = nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=3, mlp_ratio=mlp_ratio) for _ in range(num_blocks)))
        self.conv2 = ConvBNAct(2 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))
    
    def forward_split(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))

    def get_module_info(self):
        return f"VajraV1AttentionX", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.lite}]"

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.inc = inc
        self.outc = outc
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias), nn.BatchNorm2d(outc), nn.SiLU())
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = einops.rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
    
    def get_module_info(self):
        return "LDConv", f"[{self.inc}, {self.outc}, {self.num_param}, {self.stride}]"

class SwiGLUFFN(nn.Module):
    def __init__(self, in_c, out_c, mlp_ratio=4.0):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        mlp_hidden_dim = int(in_c * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.norm = LayerNorm2d(in_c)
        self.fc1 = nn.Linear(in_c, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim // 2, out_c)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        a, b = self.fc1(x).split(self.mlp_hidden_dim // 2, dim=2)
        hidden = F.silu(a) * b
        out = self.fc2(hidden)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

class Squeeze_Excite_Layer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(Squeeze_Excite_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        y = self.avg_pool(x)  # [batch_size, channels, 1, 1]
        y = self.fc(y)       # [batch_size, channels, 1, 1]
        return x * y         # Broadcasting: multiply across spatial dimensions

class VajraRepViTBlock(nn.Module):
    def __init__(self, in_c, out_c, use_se=False, stride=1, use_rep_vgg_dw = False, kernel_size=7, mlp_ratio=2.0, swiglu_ffn = True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.use_se = use_se
        hidden_c = int(mlp_ratio * in_c)
        self.add = stride == 2 or in_c == out_c

        if stride == 2:
            self.token_mixer = nn.Sequential(
                ConvBNAct(in_c, in_c, stride=2, kernel_size=3, groups=in_c, act=None),
                Squeeze_Excite_Layer(in_c, reduction=4) if use_se else nn.Identity(),
                ConvBNAct(in_c, out_c, 1, 1, act=None)
            )
            self.channel_mixer = nn.Sequential(ConvBNAct(out_c, 2 * out_c, 1, 1), ConvBNAct(2 * out_c, out_c, 1, 1, act=None))

        else:
            self.token_mixer = nn.Sequential(
                MerudandaDW(in_c, in_c, True, 0.5, use_rep_vgg_dw, kernel_size=kernel_size), #RepVGGDW(dim=in_c, kernel_size=7, stride=1),
                Squeeze_Excite_Layer(in_c, reduction=4) if use_se else nn.Identity(),
            )
            self.channel_mixer = SwiGLUFFN(in_c, out_c, 2.0) if swiglu_ffn else nn.Sequential(ConvBNAct(in_c, hidden_c, 1, 1), ConvBNAct(hidden_c, out_c, 1, 1, act=None))
        
        self.apply(self._init_weights)

    def forward(self, x):
        token_mixer_out = self.token_mixer(x)
        out = token_mixer_out + self.channel_mixer(token_mixer_out) if self.add else self.channel_mixer(token_mixer_out)
        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_module_info(self):
        return "VajraRepViTBlock", f"[{self.in_c}, {self.out_c}, {self.use_se}, {self.stride}]"

class ScalSeq(nn.Module):
    def __init__(self, in_c, channel):
        super(ScalSeq, self).__init__()
        self.in_c = in_c
        self.channels = channel
        self.conv1 =  ConvBNAct(in_c[0], channel, 1, 1)
        self.conv2 =  ConvBNAct(in_c[1], channel, 1, 1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d,p4_3d,p5_3d], dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
    
    def get_module_info(self):
        return "ScalSeq", f"[{self.in_c}, {self.channels}]"

class VajraResViTBlock(nn.Module):
    def __init__(self, in_c, out_c, use_se=False, expansion_ratio=1.0, mlp_ratio=2.0):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.use_se = use_se
        hidden_c = int(mlp_ratio * in_c)
        self.add = in_c == out_c
        
        self.token_mixer = nn.Sequential(
                BottleneckV3(in_c, in_c, True, 5, expansion_ratio=expansion_ratio),
                Squeeze_Excite_Layer(in_c, reduction=4) if use_se else nn.Identity(),
            )
        self.channel_mixer = nn.Sequential(ConvBNAct(in_c, hidden_c, 1, 1), ConvBNAct(hidden_c, out_c, 1, 1, act=None))
        
        self.apply(self._init_weights)

    def forward(self, x):
        token_mixer_out = self.token_mixer(x)
        out = token_mixer_out + self.channel_mixer(token_mixer_out) if self.add else self.channel_mixer(token_mixer_out)
        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_module_info(self):
        return "VajraResViTBlock", f"[{self.in_c}, {self.out_c}, {self.use_se}]"
    
class MLPConv(nn.Module):
    def __init__(self, in_c, out_c, mlp_ratio=1.2):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mlp_ratio = mlp_ratio
        self.add = in_c == out_c
        mlp_hidden_dim = int(in_c * mlp_ratio)
        self.mlp = nn.Sequential(ConvBNAct(in_c, mlp_hidden_dim, 1, 1), ConvBNAct(mlp_hidden_dim, 1, 1, act=None))

    def forward(self, x):
        return x + self.mlp(x) if self.add else self.mlp(x)
    
    def get_module_info(self):
        return "MLPConv", f"[{self.in_c}, {self.out_c}, {self.mlp_ratio}]"

class MLP(nn.Module):
    def __init__(self, in_c, out_c, mlp_ratio=2.0):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mlp_ratio = mlp_ratio
        self.add = in_c == out_c
        mlp_hidden_dim = int(in_c * mlp_ratio)
        self.fc1 = nn.Linear(in_c, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, out_c)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))
    
class TokenMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_ratio):
        super().__init__()
        self.num_features = num_features
        self.num_patches = num_patches
        self.expansion_ratio = expansion_ratio
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, num_patches, expansion_ratio)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H * W == self.num_patches, f"Expected {self.num_patches} patches, got {H*W}"
        x = x.flatten(2).transpose(1, 2) # B, HW, C
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        out = residual + x
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out
    
class ChannelMixer(nn.Module):
    def __init__(self, num_features, expansion_ratio):
        super().__init__()
        self.num_features = num_features
        self.expansion_ratio = expansion_ratio
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, num_features, expansion_ratio)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B, HW, C
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = residual + x
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out
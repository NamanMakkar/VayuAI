# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import numpy as np
import math
import copy
import torch
import functools
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import OrderedDict, List
from torch.nn.init import constant_, xavier_uniform_
from vajra.nn.utils import _get_clones, bias_init_with_prob
from vajra.utils import LOGGER
#from vajra.nn.transformer import LayerNorm2d
from vajra.nn.attention_utils import weighting_function, distance2bbox, deformable_attention_core_func_v2, inverse_sigmoid
from vajra.utils.dfine_ops import get_contrastive_denoising_training_group
from vajra.nn.modules import SPPF, ConvBNAct, InnerBlock, RMSNorm, DynamicTanh, MakkarNormalization, SCDown, RepNCSPELAN4, AttentionBlockV2, AttentionBlockV3, AreaAttentionBlock, act_table


def window_partition(x, window_size):
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')  # Pad right and bottom
    H_padded, W_padded = H + pad_h, W + pad_w

    x = x.reshape(B, C, H_padded // window_size, window_size, W_padded // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size, window_size, C)
    return windows, H, W

def window_reverse(windows, window_size, H_orig, W_orig):
    """B = int(windows.shape[0] / (H_orig * W_orig / (window_size * window_size)))
    t = windows.reshape(B, H_orig // window_size, W_orig // window_size, window_size, window_size, -1)
    t = t.permute(0, 5, 1, 3, 2, 4).contiguous()
    t = t.reshape(B, -1, H_orig, W_orig)"""

    B_ = windows.shape[0]
    C = windows.shape[-1]
    num_windows_h = (H_orig + (window_size - H_orig % window_size) % window_size) // window_size
    num_windows_w = (W_orig + (window_size - W_orig % window_size) % window_size) // window_size
    B = B_ // (num_windows_h * num_windows_w)

    t = windows.reshape(B, num_windows_h, num_windows_w, window_size, window_size, C)
    t = t.permute(0, 5, 1, 3, 2, 4).contiguous()
    t = t.reshape(B, C, num_windows_h * window_size, num_windows_w * window_size)
    # Crop explicitly
    t = torch.narrow(t, 2, 0, H_orig)
    t = torch.narrow(t, 3, 0, W_orig)
    return t

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop = 0., proj_drop = 0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size
        self.head_dim = dim // num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, dim, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(dim, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0) # 1, Wh - 1, Ww - 1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = ConvBNAct(dim, 3 * dim, 1, 1, act=None)
        self.proj = ConvBNAct(dim, dim, 1, 1, act=None)
        self.positional_encoding = ConvBNAct(dim, dim, 1, 3, groups=dim, act=None)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        N = self.window_size[0] * self.window_size[1]
        #B_ = (H // self.window_size[0]) * (W // self.window_size[1]) * B # num_windows * batch_size
        B_ = (H + (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]) // self.window_size[0] * \
             (W + (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]) // self.window_size[1] * B
        qkv = self.qkv(x)
        qkv_windows, H_orig, W_orig = window_partition(qkv, self.window_size[0])
        qkv_windows = qkv_windows.reshape(B_, N, self.num_heads, self.head_dim * 3)
        qkv_windows = qkv_windows.permute(0, 2, 1, 3)
        q, k, v = qkv_windows.split([self.head_dim, self.head_dim, self.head_dim], dim=-1) # B_, self.num_heads, N, self.head_dim

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) # B_, self.num_heads, N, N
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1 
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2) # B_, N, self.num_heads, self.head_dim
        x = x.reshape(B_, N, C)
        x = window_reverse(x, self.window_size[0], H_orig, W_orig) # B, C, H, W
        v = v.transpose(1, 2).reshape(B_, N, C)
        v = window_reverse(v, self.window_size[0], H_orig, W_orig)
        x = x + self.positional_encoding(v)
        x = self.proj(x)
        return x
    
class WindowAttentionSDPA(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop  # For SDPA dropout
        self.proj_drop = proj_drop

        # CPB MLP for relative position bias
        self.cpb_mlp = nn.Sequential(
            ConvBNAct(2, dim, 1, 1, bias=True),
            ConvBNAct(dim, num_heads, 1, 1, bias=False, act=None)
        )

        # Relative coords table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV and projection layers (simplified, assuming ConvBNAct → Conv2d)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.positional_encoding = nn.Conv2d(dim, dim, 7, 1, padding=3, groups=dim, bias=False)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        window_size_h, window_size_w = self.window_size
        N = window_size_h * window_size_w
        B_ = (H // window_size_h) * (W // window_size_w) * B  # num_windows * batch_size

        # QKV projection
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        qkv_windows, H_orig, W_orig = self.window_partition(qkv, window_size_h)  # (B_, Wh, Ww, 3*C)
        qkv_windows = qkv_windows.permute(0, 3, 1, 2).contiguous().view(B_, 3 * C, N)  # (B_, 3*C, N)

        # Split and reshape for multi-head attention
        qkv_windows = qkv_windows.view(B_, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv_windows[0], qkv_windows[1], qkv_windows[2]  # (B_, num_heads, N, head_dim)

        # Relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)  # (num_heads, N, N)

        # SDPA with integrated bias and mask
        if mask is not None:
            nW = mask.shape[0]  # Number of window groups (e.g., from shifted windows)
            q = q.view(B_ // nW, nW, self.num_heads, N, self.head_dim)
            k = k.view(B_ // nW, nW, self.num_heads, N, self.head_dim)
            v = v.view(B_ // nW, nW, self.num_heads, N, self.head_dim)
            # Combine mask and bias
            attn_mask = mask.unsqueeze(1) + relative_position_bias.unsqueeze(0)  # (nW, num_heads, N, N)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0
            )
            x = x.view(B_, self.num_heads, N, self.head_dim)
        else:
            attn_mask = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)  # (B_, num_heads, N, N)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0
            )

        # Reshape and reverse windows
        x = x.transpose(1, 2).contiguous().view(B_, N, C)  # (B_, N, C)
        x = self.window_reverse(x, window_size_h, H_orig, W_orig)  # (B, C, H, W)
        v = v.transpose(1, 2).contiguous().view(B_, N, C)
        v = self.window_reverse(v, window_size_h, H_orig, W_orig)
        x = x + self.positional_encoding(v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_padded, W_padded = H + pad_h, W + pad_w
        x = x.view(B, C, H_padded // window_size, window_size, W_padded // window_size, window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
        return windows, H, W

    def window_reverse(self, windows, window_size, H_orig, W_orig):
        B_ = windows.shape[0]
        C = windows.shape[-1]
        num_windows_h = (H_orig + (window_size - H_orig % window_size) % window_size) // window_size
        num_windows_w = (W_orig + (window_size - W_orig % window_size) % window_size) // window_size
        B = B_ // (num_windows_h * num_windows_w)
        t = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
        t = t.permute(0, 5, 1, 3, 2, 4).contiguous()
        t = t.view(B, C, num_windows_h * window_size, num_windows_w * window_size)
        t = torch.narrow(t, 2, 0, H_orig)
        t = torch.narrow(t, 3, 0, W_orig)
        return t
    
class VajraSwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=2., attn_drop=0., proj_drop=0., sdpa=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.norm1 = DynamicTanh(channels=dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), 
                                    num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop) if not sdpa else WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = DynamicTanh(channels = dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(ConvBNAct(dim, mlp_hidden_dim, 1, 1), ConvBNAct(mlp_hidden_dim, dim, 1, 1, act=None))
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, 1, Hp, Wp), device=x.device, dtype=x.dtype)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        mask_windows, _, _ = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def pad_to_window(self, x, window_size):
        B, C, H, W = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right and bottom
        return x, pad_h, pad_w

    def forward(self, x):
        _, _, H, W = x.shape
        self.create_mask(x, H, W)
        shortcut = x
        #x, pad_h, pad_w = self.pad_to_window(x, self.window_size)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        attn = self.attn(shifted_x, mask=self.attn_mask) # B, C, H, W

        if self.shift_size > 0:
            x = torch.roll(attn, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = attn

        #if pad_h > 0 or pad_w > 0:
            #x = x[:, :, :H, :W].contiguous()
        
        x = shortcut + self.norm1(x)
        x = x + self.norm2(self.mlp(x))
        return x
    
    def get_module_info(self):
        return "VajraSwinTransformerLayer", f"[{self.dim}, {self.num_heads}, {self.window_size}, {self.shift_size}, {self.mlp_ratio}]"

class VajraV1SwinTransformerBlockV1(nn.Module):
    def __init__(self, in_c, out_c, num_blocks = 2, expansion_ratio = 0.5, window_size=8):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.hidden_c = int(out_c * expansion_ratio)
        self.expansion_ratio = expansion_ratio

        self.conv1 = ConvBNAct(in_c, 2 * self.hidden_c, 1, 1)
        self.window_size = window_size
        self.shift_size = self.window_size // 2
        self.blocks = nn.ModuleList(
            nn.Sequential(*(VajraSwinTransformerLayer(self.hidden_c, num_heads=self.hidden_c // 32, window_size=8, shift_size = 0 if (i % 2 == 0) else self.shift_size) for i in range(2))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 2) * self.hidden_c, self.out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(attn(y[-1]) for attn in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = self.conv1(x).split((self.hidden_c, self.hidden_c), 1)
        y = [y[0], y[1]]
        y.extend(attn(y[-1]) for attn in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV1SwinTransformerBlockV1", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.window_size}]"

class VajraV1SwinTransformerBlockV2(nn.Module):
    def __init__(self, in_c, out_c, num_blocks = 2, expansion_ratio = 0.5, window_size=8, area=4):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.hidden_c = int(out_c * expansion_ratio)
        self.expansion_ratio = expansion_ratio

        self.conv1 = ConvBNAct(in_c, self.hidden_c, 1, 1)
        self.window_size = window_size
        self.shift_size = self.window_size // 2
        self.swin_attn_blocks = nn.ModuleList(
            nn.Sequential(*(VajraSwinTransformerLayer(self.hidden_c, num_heads=self.hidden_c // 32, window_size=8, shift_size = 0 if (i % 2 == 0) else self.shift_size, mlp_ratio=1.5) for i in range(2))) for _ in range(num_blocks)
        )
        self.attn_blocks = nn.ModuleList(
            nn.Sequential(*(AreaAttentionBlock(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64, kernel_size=7, mlp_ratio=1.5, area=area) for _ in range(2))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((2 * num_blocks + 1) * self.hidden_c, self.out_c, 1, 1)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(attn(y[-1]) for attn in self.swin_attn_blocks)
        y.extend(attn(y[-1]) for attn in self.attn_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV1SwinTransformerBlockV2", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.window_size}]"
    
class VajraV1SwinTransformerBlockV3(nn.Module):
    def __init__(self, in_c, out_c, num_blocks = 2, expansion_ratio = 0.5, window_size=8):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.hidden_c = int(out_c * expansion_ratio)
        self.expansion_ratio = expansion_ratio

        self.sppf = SPPF(in_c, self.hidden_c, 5)
        self.window_size = window_size
        self.shift_size = self.window_size // 2
        self.blocks = nn.ModuleList(
            nn.Sequential(*(VajraSwinTransformerLayer(self.hidden_c, num_heads=self.hidden_c // 32, window_size=8, shift_size = 0 if (i % 2 == 0) else self.shift_size) for i in range(2))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 1) * self.hidden_c, self.out_c, 1, 1)

    def forward(self, x):
        y = [self.sppf(x)]
        y.extend(attn(y[-1]) for attn in self.blocks)
        return self.conv2(torch.cat(y, 1))
    
    def get_module_info(self):
        return "VajraV1SwinTransformerBlockV3", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.window_size}]"

class VajraV1SwinTransformerBlockV4(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio = 0.5, window_size=8, mlp_ratio=1.5, use_sppf=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.hidden_c = int(out_c * expansion_ratio)
        self.expansion_ratio = expansion_ratio
        self.mlp_ratio = mlp_ratio

        self.conv1 = ConvBNAct(in_c, 2 * self.hidden_c, 1, 1) if not use_sppf else SPPF(in_c, 2 * self.hidden_c, 5)
        self.window_size = window_size
        self.shift_size = self.window_size // 2
        assert in_c == 2 * self.hidden_c, "in_c must be equal to 2 * hidden_c for fusion"
        self.swin_blocks = nn.Sequential(*(VajraSwinTransformerLayer(self.hidden_c, num_heads=self.hidden_c // 32, window_size=8, shift_size = 0 if (i % 2 == 0) else self.shift_size) for i in range(2)))
        
        self.attn_blocks = nn.ModuleList(
            nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 32, kernel_size=7, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct((num_blocks + 3) * self.hidden_c, self.out_c, 1, 1)

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [b]
        y.append(self.swin_blocks(y[-1]))
        y.extend(attn_blocks(y[-1]) for attn_blocks in self.attn_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split((self.hidden_c, self.hidden_c), 1)
        y = [b]
        y.append(self.swin_blocks(y[-1]))
        y.extend(attn_blocks(y[-1]) for attn_blocks in self.attn_blocks)
        fm = torch.cat((a, b), 1)
        fm = torch.sum(torch.stack([fm, x]), dim=0)
        y.append(fm)
        return self.conv2(torch.cat(y[1:], 1))
    
    def get_module_info(self):
        return "VajraV1SwinTransformerBlockV4", f"[{self.in_c}, {self.out_c}, {self.num_blocks}, {self.expansion_ratio}, {self.window_size}, {self.mlp_ratio}]"

class SanlayanSPPFSwinV1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, num_blocks=2, lite=False, mlp_ratio=2.0, window_size=8, expansion_ratio=0.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = int(out_c * expansion_ratio)
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.swin_attn = nn.ModuleList(
            nn.Sequential(*(VajraSwinTransformerLayer(self.hidden_c, num_heads=self.hidden_c // 32, window_size=window_size, shift_size = 0 if (i % 2 == 0) else self.shift_size, mlp_ratio=mlp_ratio) for i in range(2))) for _ in range(num_blocks)
        )
        self.attn = nn.ModuleList(
            nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=7, mlp_ratio=mlp_ratio) for _ in range(2))) for _ in range(num_blocks)
        )
        self.conv = ConvBNAct(in_c + (2 * num_blocks) * self.hidden_c, out_c, 1, 1)

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
        fms.extend(attn(fms[-1]) for attn in self.swin_attn)
        fms.extend(attn(fms[-1]) for attn in self.attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFSwinV1", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}, {self.window_size}]"
    
class SanlayanSPPFSwinV2(nn.Module):
    def __init__(self, in_c, out_c, stride=1, num_blocks=2, lite=False, mlp_ratio=2.0, window_size=8, expansion_ratio=0.5) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.num_blocks = num_blocks
        self.branch_a_channels = in_c - self.out_c
        self.hidden_c = int(out_c * expansion_ratio)
        self.out_c = out_c
        self.lite = lite
        self.sppf = SPPF(in_c=self.out_c, out_c=self.out_c, kernel_size=5)
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.swin_attn = nn.ModuleList(
            nn.Sequential(*(VajraSwinTransformerLayer(self.hidden_c, num_heads=self.hidden_c // 32, window_size=window_size, shift_size = 0 if (i % 2 == 0) else self.shift_size, mlp_ratio=mlp_ratio) for i in range(2))) for _ in range(num_blocks)
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
        fms.extend(attn(fms[-1]) for attn in self.swin_attn)
        out = self.conv(torch.cat(fms, 1))
        return out

    def get_module_info(self):
        return f"SanlayanSPPFSwinV2", f"[{self.in_c}, {self.out_c}, {self.stride}, {self.num_blocks}, {self.window_size}]"
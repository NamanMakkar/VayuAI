# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from vajra.nn.utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

from vajra.nn.modules import ConvBNAct

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        x = self.multihead_attn(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x

class TransformerBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads, num_layers) -> None:
        super().__init__()
        self.conv = None

        if in_c != out_c:
            self.conv = ConvBNAct(in_c, out_c)
        
        self.linear = nn.Linear(out_c, out_c)
        self.transformers = nn.Sequential(*(TransformerLayer(out_c, num_heads) for _ in range(num_layers)))
        self.out_c = out_c
    
    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)

        b, c, w, h = x.shape
        permuted_fm = x.flatten(2).permute(2, 0, 1)
        return self.transformers(permuted_fm + self.linear(permuted_fm)).permute(1, 2, 0).reshape(b, self.out_c, w, h)

class ScaleAdaptiveTransformerDecoder(nn.Module):
    
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        
    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = bboxes[..., :2]
        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])
            
        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist
        
        return dist
    
    def forward(
        self,
        embed,  # decoder embeddings
        refer_bbox,  # anchor
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        pos = pos_mlp(refer_bbox)
        dist = self.calc_bbox_dists(refer_bbox)
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, dist, padding_mask, attn_mask, pos)
            
            if self.training:
                dec_cls.append(score_head[i](output))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                break
            
        return torch.stack(dec_cls)
    
class ScaleAdaptiveDecoderLayer(nn.Module):
    
    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU()):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.gen_tau = nn.Linear(d_model, n_heads)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.init_weights()
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)
        
    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout2(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm2(tgt)
    
    def forward(self, embed, refer_bbox, dist, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""
        
        # Self attention
        tau = self.gen_tau(embed)  # [B, Q, 8]
        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        dist_attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]
        if attn_mask is not None:
            dist_attn_mask[:, :, attn_mask] = float('-inf')
        attn_mask = dist_attn_mask.flatten(0, 1) # [Bx8, Q, Q]
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)
        
        # FFN
        return self.forward_ffn(embed)
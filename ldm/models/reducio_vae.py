from functools import partial
import time
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from torch import nn, einsum
from einops import rearrange, repeat
from ldm.util import compute_ssim, compute_psnr, default, instantiate_from_config
from ldm.modules.diffusionmodules.model import make_conv, nonlinearity, ResnetBlock, Normalize, checkpoint_new
from taming.modules.losses.lpips import LPIPS
import pytorch_lightning as pl
from itertools import repeat
from torch.optim.lr_scheduler import LambdaLR
import math
import random

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")

def exists(val):
    return val is not None

def get_sinusoid_encoding_table(n_position, d_hid, device='cpu'): 
    ''' Sinusoid position encoding table ''' 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    return  torch.tensor(sinusoid_table, requires_grad=False).unsqueeze(0).to(device)


class UpsampleUnshuffle(nn.Module):
    def __init__(self, in_channels, keep_temp=False, keep_spatial=False, mode='2d', use_act= False):
        super().__init__()
        assert not (keep_spatial and keep_temp)
        self.keep_temp = keep_temp
        self.keep_spatial = keep_spatial
        self.mode = mode
        out_channels = in_channels *2 if self.keep_spatial  else in_channels *4  if self.keep_temp or mode =='2d' else in_channels * 8
        self.conv = make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode=mode)
        self.act_func = nn.SiLU() if use_act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act_func(x)
        if self.mode != '2d':
            _, c, t, h, w = x.shape
            if self.keep_temp:
                p1, p2, p3 = 1, 2, 2
            elif self.keep_spatial:
                p1, p2, p3 = 2, 1, 1
            else:
                p1, p2, p3 = 2, 2, 2
            x = rearrange(x, "b (d p1 p2 p3) t h w -> b d (t p1) (h p2) (w p3)", p1=p1, p2=p2, p3=p3)
        else:
            x = rearrange(x, "b (d p1 p2) h w -> b d (h p1) (w p2)", p1=2, p2=2)
        return x
    
        
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, out_channels=None, keep_temp=False, keep_spatial=False, mode='2d'):
        super().__init__()
        assert not (keep_spatial and keep_temp)
        out_channels = in_channels if out_channels is None else out_channels
        self.with_conv = with_conv
        self.keep_temp = keep_temp
        self.keep_spatial = keep_spatial
        if self.with_conv:
            self.conv = make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode=mode)
    
    def forward(self, x):
        if self.keep_temp:
            x = torch.nn.functional.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        elif self.keep_spatial:
            x = torch.nn.functional.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="nearest")
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, mode='2d', keep_temp=False, keep_spatial=False):
        super().__init__()
        self.with_conv = with_conv
        self.mode = mode
        self.keep_temp = keep_temp
        self.keep_spatial = keep_spatial
        
        if self.with_conv and mode == '2d':
            kernel_size = 3
            stride =2
            self.padding = (0,1,0,1)
            self.conv = make_conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=(0,0), mode='2d')
        
        elif with_conv:
            if keep_temp:
                kernel_size = (1,3,3)
                stride = (1, 2, 2)
            elif keep_spatial:
                kernel_size=(3, 1, 1)
                stride=(2, 1, 1)
            else:
                kernel_size = (3,3,3)
                stride = (2, 2, 2)
            padding = (0, 0, 0)
            self.conv = make_conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, mode='3d')
            if keep_temp:
                self.padding = (0,1,0,1,0,0)
            elif keep_spatial and mode == 'causal3d':
                self.padding = (0,0,0,0,1,0)
            elif keep_spatial:
                self.padding = (0,0,0,0,0,1)   
            elif mode == 'causal3d':
                self.padding = (0,1,0,1,1,0)
            else:
                self.padding = (0,1,0,1,0,1)
                
    def forward(self, x):
        if self.with_conv:
            x = torch.nn.functional.pad(x,  self.padding, mode="constant", value=0)
            x = self.conv(x)
        else:
            if self.mode != '2d' and self.keep_temp:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            elif self.mode != '2d' and self.keep_spatial:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            elif self.mode != '2d':
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def window_partition(x, window_size, retain_shape=False):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if x.ndim == 4:
        if retain_shape:
            return rearrange(x, "b c (h w1) (w w2) ->  (b h w) c w1 w2", w1=window_size[0], w2=window_size[1])
        else:
            return rearrange(x, "b c (h w1) (w w2) ->  (b h w) (w1 w2) c", w1=window_size[0], w2=window_size[1])
    else:
        if retain_shape:
            return rearrange(x, "b c t (h w1) (w w2) ->  (b h w) c t w1 w2", w1=window_size[0], w2=window_size[1])
        else:
            return rearrange(x, "b c t (h w1) (w w2) ->  (b h w) (t w1 w2) c", w1=window_size[0], w2=window_size[1])
        

def window_reverse(x, window_size, H, W, volumetric=True):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    if volumetric:
        return rearrange(x, "(b h w) (t w1 w2) c  -> b c t (h w1) (w w2)",  h=H // window_size[0], w=W // window_size[1], w1=window_size[0], w2=window_size[1])
    else:
        return rearrange(x, "(b h w) (w1 w2) c  -> b c (h w1) (w w2)",  h=H // window_size[0], w=W // window_size[1], w1=window_size[0], w2=window_size[1])


class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels, mode='2d', num_heads=8, window_size =(32, 32), temp_res=None, use_causal_mask=False, fp32_attention=True):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.window_size = window_size
        
        conv_mode = '2d' if mode =='2d' else '3d'
        self.volumetric = False  if mode =='2d' else True
        self.fp32_attention = fp32_attention
        self.q = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=conv_mode)
        self.k = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=conv_mode)
        self.v = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=conv_mode)
        self.proj_out = make_conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, mode=conv_mode)
        self.num_heads = num_heads

        self.use_causal_mask = use_causal_mask and '3d' in conv_mode
        self.temp_res = temp_res
        if use_causal_mask and '3d' in conv_mode:
            assert temp_res != None
            mask = np.tril(np.ones((temp_res, temp_res)))
            mask[mask==0.] =  float("-inf")
            mask = torch.from_numpy(mask)
            self.register_buffer("attn_mask", mask)
            
            
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        enable_window_attn = False
        if self.volumetric:
            B, C, T, H, W = h_.shape
            if H * W > self.window_size[0] * self.window_size[1]:
                enable_window_attn = True
                pad_r_3d = (self.window_size_[1] - W % self.window_size_[1]) % self.window_size_[1]
                pad_b_3d = (self.window_size_[0] - H % self.window_size_[0]) % self.window_size_[0]
                h_ = F.pad(h_, (0, pad_r_3d, 0, pad_b_3d))
                _, _, _, Hp, Wp = h_.shape
                h_ = window_partition(h_, self.window_size, True)
        else:
            B, C, H, W = x.shape
            if H * W > self.window_size[0] * self.window_size[1]:
                enable_window_attn = True
                pad_r_2d = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
                pad_b_2d = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
                h_ = F.pad(h_, (0, pad_r_2d, 0, pad_b_2d))
                _, _, Hp, Wp = h_.shape
                h_ = window_partition(h_, self.window_size, True)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if self.volumetric:
            q, k, v = map(lambda x: rearrange(x, 'b c t h w -> b (t h w) c').contiguous(), (q, k, v))
        else:
            q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c').contiguous(), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q, k, v))
        
        out = xformers.ops.memory_efficient_attention(q, k, v,  attn_bias=None)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads).contiguous()
            
        if self.volumetric:
            if enable_window_attn:
                out = window_reverse(out, self.window_size, Hp, Wp, True)  
                if pad_r_3d > 0 or pad_b_3d > 0:
                    out = out[:, :, :, :H, :W].contiguous()
            else:
                out = rearrange(out, 'b (t h w) c -> b c t h w', b=B, h=H, w=W, c=C).contiguous()
        else:
            if enable_window_attn:
                out = window_reverse(out, self.window_size, Hp, Wp, False)  
                if pad_r_2d > 0 or pad_b_2d > 0:
                    out = out[:, :, :H, :W].contiguous()
            else:
                out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C).contiguous()
            
        out = self.proj_out(out)
        
        return x + out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, window_size =(64, 64), context_dim=None, qkv_bias=False, heads=8, dim_head=64, dropout=0., use_checkpoint = False, fp32_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.use_checkpoint = use_checkpoint
        self.heads = heads
        self.window_size = window_size
        self.fp32_attention = fp32_attention
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def _forward(self, x, context):
        
        context = default(context, x)

        enable_window_attn = False
        
        b, _, T, H, W = x.shape
        H_, W_ = context.shape[-2],  context.shape[-1]
        
        if H_ * W_ > self.window_size[0] * self.window_size[1] and H * W >= self.window_size[0] * self.window_size[1]:
            enable_window_attn = True
            pad_r_2d = (self.window_size[1] - W_ % self.window_size[1]) % self.window_size[1]
            pad_b_2d = (self.window_size[0] - H_ % self.window_size[0]) % self.window_size[0]
            context = F.pad(context, (0, pad_r_2d, 0, pad_b_2d))
            
            window_size_ = (math.ceil(float(self.window_size[0]) * H / H_ ), math.ceil(float(self.window_size[1]) * W / W_ ))
            pad_r_3d = (window_size_[1] - W % window_size_[1]) % window_size_[1]
            pad_b_3d = (window_size_[0] - H % window_size_[0]) % window_size_[0]
            x = F.pad(x, (0, pad_r_3d, 0, pad_b_3d))
            _, _, _, Hp, Wp = x.shape
            
            context = window_partition(context, self.window_size)
            x = window_partition(x, window_size_)
            B_w = x.shape[0]
            N_w_3d = x.shape[1]
            N_w_2d = context.shape[1]
        else:
            x = rearrange(x, 'b d t h w-> b (t h w) d').contiguous()
            context = rearrange(context, 'b d h w-> b (h w) d').contiguous()
            enable_window_attn = False
            
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        out = xformers.ops.memory_efficient_attention(q, k, v,  attn_bias=None)
        
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()
        
        if enable_window_attn:
            out = window_reverse(out, window_size_, Hp, Wp)  # B H' W' C        
            if pad_r_3d > 0 or pad_b_3d > 0:
                out = out[:, :, :, :H, :W].contiguous()
            out = rearrange(out, " b c t h w -> b (t h w) c")
            
        return self.to_out(out)
    
    def forward(self, x, context):
        if self.use_checkpoint and self.training:
            return checkpoint_new(self._forward, (x, context), self.parameters())
        else:
            return self._forward(x, context)
  
        
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, temp_res, spatial_res, d_head=64, dropout=0., use_checkpoint=False, is_tanh_gating=False, init_values=0., 
                 norm_type='layer', window_size = (32, 32), enable_window_attn = False, fp32_attention=True, pos_embed_mode = 't', use_conv_shortcut = False):
        super().__init__()
        n_heads = int(dim/d_head) if int(dim/d_head) >= 1 else 1
        self.attn = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, window_size = window_size,
                                   dropout=dropout, use_checkpoint=use_checkpoint,  fp32_attention=fp32_attention)  
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(dim) 
        else:
            self.norm = Normalize(dim) if dim > 32 else nn.BatchNorm3d(dim)
        
        self.is_tanh_gating =is_tanh_gating
        self.pos_embed_mode = pos_embed_mode
        self.norm_type = norm_type
        self.use_conv_shortcut = use_conv_shortcut
        self.context_dim = context_dim
        
        if is_tanh_gating:
            self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        if pos_embed_mode == "t":
            self.pos_embed = nn.Parameter(torch.zeros(1, temp_res, 1, dim), requires_grad=False)
            pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], int(self.pos_embed.shape[-1]))
            self.pos_embed.data.copy_(pos_embed.unsqueeze(2))
        elif pos_embed_mode == 'thw':
            self.pos_embed = nn.Parameter(torch.zeros(1, spatial_res*spatial_res*temp_res, dim), requires_grad=False)
            pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], int(self.pos_embed.shape[-1]))
            self.pos_embed.data.copy_(pos_embed)
        else:
            self.pos_embed = None
        
        if use_conv_shortcut:
            self.conv_shortcut = nn.Conv3d(dim, dim,kernel_size=3,stride=1, padding=1)
            nn.init.zeros_(self.conv_shortcut.bias)
            nn.init.dirac_(self.conv_shortcut.weight)
            
    def forward(self, x, context):
        b, _, t, h, w = x.size()
        _, d, h1, w1 = context.size()
        
        if self.norm_type== 'layer':
            x = rearrange(x, 'b d t h w-> b (t h w) d').contiguous()
            context = rearrange(context, 'b d h w-> b (h w) d').contiguous()
           
            if self.pos_embed_mode == 't':
                pos_embed = self.pos_embed.expand(x.shape[0], -1, h*w,-1)
                x = x + pos_embed.reshape(x.shape[0], t*h*w, -1)
            
            elif self.pos_embed_mode == 'thw':
                x = x + self.pos_embed.expand(x.shape[0], -1, -1)
            
            
            if self.is_tanh_gating:
                out = self.gamma * self.attn(rearrange(self.norm(x), "b (t h w) d ->  b d t h w", t=t, h=h).contiguous(),  rearrange(context, "b (h w) d ->  b d h w", h=h1).contiguous())
            else:
                out =  self.attn(rearrange(self.norm(x), "b (t h w) d ->  b d t h w", t=t, h=h).contiguous(),  rearrange(context, "b (h w) d ->  b d h w", h=h1).contiguous())
            
            out = rearrange(out, 'b (t h w) d-> b d t h w', t=t, w=w).contiguous()
            if self.use_conv_shortcut:
                return  self.conv_shortcut(rearrange(x, 'b (t h w) d-> b d t h w', t=t, w=w).contiguous()) + out
            else:
                return rearrange(x, 'b (t h w) d-> b d t h w', t=t, w=w).contiguous() + out
        else:
            if self.pos_embed_mode == 't':
                pos_embed = self.pos_embed.expand(x.shape[0], -1, h*w,-1)
                x = x + rearrange(pos_embed, "b t (h w) c -> b c t h w", h=h)
            elif self.pos_embed_mode == 'thw':
                x = x + rearrange(pos_embed, " b (t h w) c -> b c t h w", t=t, h=h)
            
            out = self.attn(self.norm(x), context)
            if self.is_tanh_gating:
                out = self.gamma * out
            if self.use_conv_shortcut:
                return  self.conv_shortcut(x) +  rearrange(out, 'b (t h w) d-> b d t h w', t=t, w=w).contiguous()
            else:
                return x + rearrange(out, 'b (t h w) d-> b d t h w', t=t, w=w).contiguous()


class LinearFusion(nn.Module):
    def __init__(self, dim, context_dim, temp_dim, dropout=0., use_skip=True, zero_init=True):
        super().__init__()

        self.conv_context = nn.Conv2d(context_dim, dim, 3, 1, 1)
        self.conv = nn.Conv3d(temp_dim + 1, temp_dim, 1)
        self.norm = Normalize(dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.use_skip = use_skip
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv_context.weight)
            nn.init.zeros_(self.conv.bias)
            nn.init.zeros_(self.conv_context.bias)
            nn.init.zeros_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x, context):
        context = self.conv_context(context).unsqueeze(2)
        out = self.conv(torch.cat([x, context], dim=2).transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        out = self.norm(out)
        out = nonlinearity(out)
        out = self.dropout(out)
        
        if self.use_skip:
            return x + out
        else:
            return out


class AddFusion(nn.Module):
    def __init__(self, dim, context_dim, temp_dim, dropout=0., use_skip=True, zero_init=True, mode='3d'):
        super().__init__()

        self.conv_context = nn.Conv2d(context_dim, dim, 3, 1, 1)
        self.temp_embed = nn.Parameter(torch.zeros(1, dim, temp_dim, 1, 1))
        self.conv = make_conv(dim, dim, 3, 1, 1, mode=mode)
        self.norm = Normalize(dim)
        self.dropout = torch.nn.Dropout(dropout)
        nn.init.trunc_normal_(self.temp_embed, std=.02)
        self.use_skip = use_skip
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv_context.weight)
            nn.init.zeros_(self.conv.bias)
            nn.init.zeros_(self.conv_context.bias)
            nn.init.zeros_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
            
    def forward(self, x, context):
        b, c, t, h, w = x.shape
        context = self.conv_context(context).unsqueeze(2)
        
        out = self.conv(x + context + self.temp_embed.expand(b, -1, -1, h, w).contiguous()) #.transpose(2, 1).transpose(2, 1)
        out = self.norm(out)
        out = nonlinearity(out)
        out = self.dropout(out)
        if self.use_skip:
            return x + out
        else:
            return out 


class Encoder(nn.Module):
    def __init__(self, *, 
                 ch, 
                 ch_mult=(1,2,4,8), 
                 num_res_blocks,
                 attn_resolutions, 
                 dropout=0.0, 
                 in_channels,
                 resolution, 
                 z_channels, 
                 out_z=True, 
                 temp_res=16,
                 use_3d_conv=False, 
                 mode='2d', 
                 f_t=0,
                 f_s=None,
                 resamp_with_conv=True, 
                 use_checkpoint=False,
                 fp32_attention=True,
                 double_z=True,
                 window_size=(256, 256),
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_z = out_z
        self.mode = mode
        self.use_3d_conv = use_3d_conv and mode != '2d' 
        

        if isinstance(window_size, int):
            window_size = tuple(repeat(window_size, 2))

        self.conv_in = make_conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1, mode=mode) 
        f_s = default(f_s, self.num_resolutions - 1)
        
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            downsample = nn.Identity()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout,
                                         temb_channels=0,
                                         use_checkpoint=use_checkpoint, 
                                         mode=mode))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            if  i_level != self.num_resolutions-1 and i_level >= self.num_resolutions- 1 - max(f_s, f_t):
                downsample = Downsample(block_in, resamp_with_conv, mode, f_s > f_t and i_level < self.num_resolutions - f_t - 1, f_t > f_s and i_level < self.num_resolutions - f_s - 1)
  
            down.downsample = downsample
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=dropout,
                                       use_checkpoint=use_checkpoint, 
                                       mode=mode,
                                       )
        attn_kwargs = {'temp_res': temp_res // 2**f_t,  'use_causal_mask': 'causal' in mode, 'fp32_attention': fp32_attention, 'window_size':window_size}
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in, mode=mode, **attn_kwargs)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=dropout,
                                       use_checkpoint=use_checkpoint, 
                                       mode=mode,)

        # end
        self.norm_out = Normalize(block_in)
        if out_z:
            self.conv_out = make_conv(block_in, 
                                    2*z_channels if double_z else z_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    mode=mode,
                                    )
        else:
            self.conv_out = nn.Identity()

    def forward(self, x, return_skip=False):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        hs = []

        for i_level in range(len(self.down)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            hs.append(h)
            h = self.down[i_level].downsample(h)

        is_video = h.dim() == 5
        
        if is_video and not self.use_3d_conv:
            h = rearrange(h, "b c t h w -> (b t) c h w")
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        if not self.out_z: 
            hs.append(h)
        
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.out_z: 
            hs.append(h)
        if return_skip:
            return hs
        else:
            return h


class FusionDecoder(nn.Module):
    def __init__(self, 
        *, 
        ch, 
        ch_2d, 
        out_ch, 
        num_res_blocks,
        attn_resolutions, 
        in_channels,
        resolution, 
        temp_res,
        z_channels, 
        z_channels_2d,
        f_t = 0, 
        f_s=None,
        ch_in=None,
        ch_mult=(1,2,4,8), 
        ch_fuse=(-1,-1,-1,-1),
        ch_mult_2d=(1,2,4,8),
        f_s_2d=None,
        out_z_2d = False,
        dropout=0.0, 
        resamp_with_conv=True, 
        tanh_out=False, 
        fuse_type=None,
        use_3d_conv=True, 
        upsample_mode='interpolate',
        upsample_first=True,
        fuse_mid=False,
        fuse_mid_type=None,
        use_checkpoint=False,
        enable_fuse=True,
        pos_embed_mode='none',
        use_conv_shortcut=False,
        is_tanh_gating=False,
        init_values=0.,
        norm_type='layer',
        mode='3d',
        window_size=(256, 256),
        fp32_attention = True,
        **ignorekwargs
    ):
        super().__init__()
        
        assert isinstance(fuse_type, str) or fuse_type is None or len(fuse_type) == len(ch_mult)
        if isinstance(fuse_type, str) or fuse_type is None:
            fuse_type = [fuse_type] * len(ch_mult)
        self.ch = ch_2d
        self.temb_ch = 0
        self.num_resolutions =  len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out
        self.ch_fuse = ch_fuse
        self.upsample_first = upsample_first
        self.fuse_mid = fuse_mid
        self.enable_fuse = enable_fuse
                
        if isinstance(window_size, int):
            window_size = tuple(repeat(window_size, 2))
            
        self.use_3d_conv = use_3d_conv
        f_s = default(f_s, self.num_resolutions - 1)
        f_s_2d = default(f_s_2d, len(ch_mult_2d) - 1)
        in_ch_mult = (1,)+tuple(ch_mult)
        ch_in = default(ch_in, ch)
        
        block_in_m = ch * ch_mult[-1]
        block_in = ch_in * ch_mult[-1] 
        
        curr_res = resolution // 2**(f_s)
        curr_temp_res = temp_res // 2**(f_t)
        
        self.z_shape = (1,z_channels, curr_temp_res, curr_res, curr_res)
        
        print(
            "Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = make_conv(z_channels, block_in_m, kernel_size=3, stride=1, padding=1, mode=mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in_m,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
            use_checkpoint=use_checkpoint, 
            mode=mode,
        )
        
        self.mid.fuse = nn.ModuleList()
        self.out_z_2d = out_z_2d
        if fuse_mid:
            context_dim = z_channels_2d*2 if out_z_2d else ch_2d*ch_mult_2d[-1]
            motion_dim = z_channels if out_z_2d else block_in
            d_head = context_dim  if out_z_2d else 64
            if f_s != f_s_2d or fuse_mid_type == 'attn':
                self.mid.fuse.append(BasicTransformerBlock(motion_dim, context_dim, temp_res // 2**f_t, resolution // 2**f_s, use_checkpoint=use_checkpoint, d_head=d_head, \
                    pos_embed_mode=pos_embed_mode, use_conv_shortcut=use_conv_shortcut, is_tanh_gating=is_tanh_gating, init_values=init_values, norm_type=norm_type, fp32_attention=fp32_attention, window_size=window_size))
            elif fuse_mid_type == 'linear':
                self.mid.fuse.append(LinearFusion(motion_dim, context_dim, temp_res // 2**f_t)) 
            elif fuse_mid_type == 'add':
                self.mid.fuse.append(AddFusion(motion_dim, context_dim, temp_res // 2**f_t))
            elif fuse_mid_type == 'add+attn':
                self.mid.fuse.append(AddFusion(motion_dim, context_dim,  temp_res // 2**f_t))
                self.mid.fuse.append(BasicTransformerBlock(motion_dim, context_dim, temp_res // 2**f_t, resolution // 2**f_s, use_checkpoint=use_checkpoint, d_head=d_head,    \
                    pos_embed_mode=pos_embed_mode, use_conv_shortcut=use_conv_shortcut, is_tanh_gating=is_tanh_gating, init_values=init_values, norm_type=norm_type, fp32_attention=fp32_attention, window_size=window_size))
            print(f"fuse on middle block w/. {fuse_mid_type}, depth: {temp_res // 2**f_t}, block_in: {motion_dim}, context_dim: {context_dim}")

        attn_kwargs = {'temp_res': temp_res // 2**f_t,  'use_causal_mask': 'causal' in mode, 'fp32_attention': fp32_attention, 'window_size': window_size}
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in, mode=mode, **attn_kwargs)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
            use_checkpoint=use_checkpoint, 
            mode=mode
        )
        
    
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_1 = nn.ModuleList()
            block_2 = nn.ModuleList()
            upsample = nn.Identity()
            fuse = nn.ModuleList()
            block_out = ch_in * ch_mult[i_level] # TODO: 
            for i_block in range(self.num_res_blocks):
                block_1.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=dropout,
                                         use_checkpoint=use_checkpoint, 
                                         mode='3d')) # if i_level == 1 else '3dHD'))
                block_in = block_out
            
            if ch_fuse[i_level] >= 0:
                fuse_offset = 0 if upsample_first else 1
                if i_level  <  self.num_resolutions - f_t - fuse_offset: # (6-1)
                    depth = temp_res
                elif i_level == self.num_resolutions - 1:
                    depth =  temp_res // 2**f_t
                else:
                    depth =  temp_res // 2**(f_t - self.num_resolutions + 1  + i_level + fuse_offset)
                
                context_dim = ch_2d*ch_mult_2d[ch_fuse[i_level]] 
                context_spatial_dim = resolution // 2** ch_fuse[i_level] 
                spatial_dim = resolution // 2**(i_level + fuse_offset) if i_level <=  f_s - fuse_offset else 2**f_s
                print(f"fuse on up {i_level}  w/. {fuse_type[i_level]}, motion_dim: {block_in},{depth},{spatial_dim},{spatial_dim}, context_dim:{context_dim},{context_spatial_dim},{context_spatial_dim}") 
                
                if fuse_type[i_level] == 'linear':
                    fuse.append(LinearFusion(block_in, context_dim, depth)) 
                elif fuse_type[i_level] == 'attn':
                    fuse.append(BasicTransformerBlock(block_in, context_dim, depth, spatial_dim, use_checkpoint=use_checkpoint,   \
                    pos_embed_mode=pos_embed_mode, use_conv_shortcut=use_conv_shortcut, is_tanh_gating=is_tanh_gating, init_values=init_values, norm_type=norm_type, fp32_attention=fp32_attention, window_size=window_size))
                elif fuse_type[i_level] == 'add':
                    fuse.append(AddFusion(block_in, context_dim,  depth))
                elif fuse_type[i_level] == 'add+attn':
                    fuse.append(AddFusion(block_in, context_dim,  depth))
                    fuse.append(BasicTransformerBlock(block_in, context_dim, depth, spatial_dim, use_checkpoint=use_checkpoint,   \
                    pos_embed_mode=pos_embed_mode, use_conv_shortcut=use_conv_shortcut, is_tanh_gating=is_tanh_gating, init_values=init_values, norm_type=norm_type, fp32_attention=fp32_attention, window_size=window_size))
                else:
                    raise RuntimeError(f"Unkonwn fusion type {fuse_type[i_level]}")
        
            for i_block in range(self.num_res_blocks):
                block_2.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_in,
                                         temb_channels=0,
                                         dropout=dropout,
                                         use_checkpoint=use_checkpoint,
                                         mode='3d' )) 
            
            if (upsample_first and i_level!= 0) or (not upsample_first and i_level != self.num_resolutions - 1):
                upsample_offset = 0 if upsample_first else 1
                if i_level >= self.num_resolutions - max(f_s, f_t) - upsample_offset:
                    if upsample_mode == 'interpolate':
                        upsample = Upsample(block_in, resamp_with_conv, block_in, f_t < f_s and i_level < self.num_resolutions - f_t - upsample_offset, f_s < f_t and i_level < self.num_resolutions - f_s - upsample_offset, mode=mode) 
                    elif upsample_mode == 'unshuffle':
                        upsample = UpsampleUnshuffle(block_in,  f_t < f_s and i_level < self.num_resolutions - f_t - upsample_offset,   f_s < f_t and i_level < self.num_resolutions - f_s - upsample_offset, mode=mode)
                    else:
                        raise NotImplementedError

            up = nn.Module()
            up.block_1 = block_1
            up.fuse = fuse
            up.block_2 = block_2
            up.upsample = upsample
            self.up.insert(0, up) 

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, content):
        bs  = z.shape[0]
        self.last_z_shape = z.shape
        
        contents = []
        for i_level in range(self.num_resolutions):
            if self.ch_fuse[i_level] != -1 and content is not None:
                contents.append(content[self.ch_fuse[i_level]].contiguous())
            else:
                contents.append(0)

        temb = None
        
        if len(self.mid.fuse) > 0 and self.out_z_2d and self.enable_fuse and content is not None:
            if not self.use_3d_conv:
                z = rearrange(z, "(b t) c h w -> b c t h w", b=bs)
            for i in range(len(self.mid.fuse)):
                z = self.mid.fuse[i](z, content[-1].contiguous())
            if not self.use_3d_conv:
                z = rearrange(z, " b c t h w-> (b t) c h w ")
                
        m = self.conv_in(z)
        m = self.mid.block_1(m)
        
        
        if len(self.mid.fuse) > 0 and not self.out_z_2d  and self.enable_fuse and content is not None:
            if not self.use_3d_conv:
                m = rearrange(m, "(b t) c h w -> b c t h w", b=bs)
            for i in range(len(self.mid.fuse)):
                m = self.mid.fuse[i](m, content[-1].contiguous())
            if not self.use_3d_conv:
                m = rearrange(m, " b c t h w-> (b t) c h w ")

        m = self.mid.attn_1(m)
        m = self.mid.block_2(m)
        
        if not self.use_3d_conv:
            h = rearrange(m, "(b t) c h w -> b c t h w", b=bs)
        else:
            h = m
        
        for i_level in reversed(range(len(self.up))):
            if len(self.up[i_level].block_1) > 0:
                for i_block in range(self.num_res_blocks):
                    h = self.up[i_level].block_1[i_block](h)
            
            if len(self.up[i_level].fuse) > 0 and self.enable_fuse and content is not None:
                context = contents[i_level]
                for i in range(len(self.up[i_level].fuse)):
                    h = self.up[i_level].fuse[i](h, context.contiguous())
                     
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block_2[i_block](h)
            
            h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
            
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.is_video = self.mean.ndim == 5
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.dtype = parameters.dtype
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, dtype=self.dtype).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None and self.is_video:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3, 4])
            elif other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

   
class ReducioVAE(pl.LightningModule):
    """
    Base class for standard and residual UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, 
                 lossconfig, 
                 ddconfig, 
                 ddconfig_2d,
                 embed_dim, 
                 ckpt_path=None, 
                 ignore_keys=[], 
                 monitor=None,
                 image_key="image",
                 automatic_optimization = False,
                 gradient_clip_val=None,
                 gradient_clip_val_disc=None,
                 mask_config=None,
                 scheduler_config=None, 
                 scheduler_disc_config = None,
                 enable_2d=True,
                 freeze_2d = False,
                 freeze_3d = False,
                 tile_sample_min_size = 256,
                 use_tiling = False,
                 tile_overlap_factor = 0.2,
                 mode = 'full',
                 **kwargs):
        super().__init__()
        # assert len(ch_mult_3d) == len(ch_fuse)
        self.loss = instantiate_from_config(lossconfig) 
        self.image_key = image_key
        self.enable_2d = enable_2d
        self.freeze_2d = freeze_2d
        self.freeze_3d = freeze_3d
        self.disc_loss_scale = self.loss.disc_loss_scale if self.loss is not None else 1.0 #default(lossconfig['params']['disc_loss_scale'], 1.0)
        self.mode = mode
        self.automatic_optimization = automatic_optimization

        conv_mode = 'causal3d' if mode =='causal' else '3d'
        self.encoder_3d = Encoder(mode=conv_mode, out_z=True,  **ddconfig)
        if enable_2d:
            self.encoder_2d = Encoder(mode='2d', **ddconfig_2d)
        
        self.decoder = FusionDecoder(
                ch_2d=ddconfig_2d["ch"], 
                ch_mult_2d=ddconfig_2d["ch_mult"], 
                out_z_2d=ddconfig_2d['out_z'],
                z_channels_2d = ddconfig_2d['z_channels'],
                f_s_2d=ddconfig_2d['f_s'] if 'fs' in ddconfig_2d else None,
                mode = conv_mode,
                **ddconfig)
        
        self.time_downsample_factor = 2 ** ddconfig["f_t"]
        self.num_frames = ddconfig["temp_res"]
        self.enable_fuse = ddconfig.get('enable_fuse', True) and self.enable_2d

        self.use_3d_conv = ddconfig['use_3d_conv'] #and 2**ddconfig["f_t"] < ddconfig["temp_res"]
        self.quant_conv = make_conv(2*ddconfig["z_channels"], 2*embed_dim, 1, stride =1, mode=conv_mode)
        self.post_quant_conv = make_conv(embed_dim, ddconfig["z_channels"], 1,  stride =1, mode=conv_mode)
        
        self.embed_dim = embed_dim
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_val_disc = gradient_clip_val_disc

        self.scheduler_disc_config = scheduler_disc_config
        self.scheduler_config = scheduler_config
        
        self.use_scheduler = scheduler_config is not None
        self.use_scheduler_disc = scheduler_disc_config is not None
        
        self.fs_3d  = ddconfig['f_s'] if 'f_s' in ddconfig else len(ddconfig['ch_mult']) - 1
        self.ft_3d  = ddconfig['f_t'] if 'f_t' in ddconfig else len(ddconfig['ch_mult']) - 1
        self.ch_fuse = ddconfig['ch_fuse']
        
        self.use_tiling = use_tiling
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_overlap_factor = tile_overlap_factor
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** self.fs_3d ))
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze_2d:
            for param in self.encoder_2d.parameters():
                param.requires_grad = False
        if freeze_3d:
            for param in self.encoder_3d.parameters():
                param.requires_grad = False
            self.quant_conv.weight.requires_grad = False
            self.quant_conv.bias.requires_grad = False
                
                
        print(f"{self.__class__.__name__} Model Parameters: {sum(p.numel() for p in self.parameters()):,}")
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        if list(sd.keys())[0].startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in sd.items():
                name = k[7:]
                new_state_dict[name] = v
            sd = new_state_dict
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k and k in sd:
                    # if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(msg)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def blend_v(self, a, b, blend_extent: int) -> torch.Tensor:
        if isinstance(a, (tuple, list)):
            for i in range(len(a)):
                if b[i] is not None:
                    blend_extent_ = int(a[i].shape[2] / (1 + self.tile_overlap_factor)  * self.tile_overlap_factor)
                    blend_extent_ = min(a[i].shape[2], b[i].shape[2], blend_extent_)
                    for y in range(blend_extent_):
                        b[i][:, :, y, :] = a[i][:, :, -blend_extent_ + y, :] * (1 - y / blend_extent_) + b[i][:, :, y, :] * (y / blend_extent_)
        else:
            blend_extent = min(a.shape[3], b.shape[3], blend_extent)
            for y in range(blend_extent):
                b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b
    
    def blend_t(self, a, b, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :, :] = a[:, :, -blend_extent + y, :, :] * (1 - y / blend_extent) + b[:, :, y, :, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent: int) -> torch.Tensor:
        if isinstance(a, (tuple, list)):
            for i in range(len(a)):
                if b[i] is not None:
                    blend_extent_ = int(a[i].shape[3] / (1 + self.tile_overlap_factor)  * self.tile_overlap_factor)
                    blend_extent_ = min(a[i].shape[3], b[i].shape[3], blend_extent_)
                    for x in range(blend_extent_):
                        b[i][:, :, :, x] = a[i][:, :, :, -blend_extent_ + x] * (1 - x / blend_extent_) + b[i][:, :, :, x] * (x / blend_extent_)
        else:
            blend_extent = min(a.shape[4], b.shape[4], blend_extent)
            for x in range(blend_extent):
                b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b
    
    def tiled_encode_2d(self, x):
        # tilding_stride = self.tile_sample_min_size
        tilding_stride = int(self.tile_sample_min_size * (1 + self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        
        rows = []
        for i in range(0, x.shape[2], self.tile_sample_min_size):
            row = []
            for j in range(0, x.shape[3], self.tile_sample_min_size):
                tile = x[:, :, i : i + tilding_stride, j : j + tilding_stride]
                tile = self.encoder_2d(tile, True)
                row.append(tile)
            rows.append(row)
        
        # return rows
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                res_row = []
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile)
            result_rows.append(result_row)
        return result_rows

    def encode_2d(self, x, use_tiling=False):
        assert len(x.shape) == 4
        if use_tiling:
            h = self.tiled_encode_2d(x)
        else:
            h = self.encoder_2d(x, return_skip=True)
        return h

    def tiled_encode_3d(self, x):
        tilding_stride = int(self.tile_sample_min_size * (1 + self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size 

        rows = []
        for i in range(0, x.shape[3], self.tile_sample_min_size):
            row = []
            for j in range(0, x.shape[4], self.tile_sample_min_size):
                tile = x[:, :, :, i : i + tilding_stride, j : j + tilding_stride]
                tile = self.encoder_3d(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))
        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
        
    def encode_3d(self, x, use_tiling=False):
        bs = x.shape[0]
        if use_tiling:
            return  self.tiled_encode_3d(x)
        else:
            x = self.encoder_3d(x)
            moments = self.quant_conv(x)
            if not self.use_3d_conv:
                moments = rearrange(moments, "(b t)  c h w  ->b c t h w", b=bs)
            posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def tiled_decode(self, z, x0_feats):
        tilding_stride = int(self.tile_latent_min_size * (1 + self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        
        row_limit = self.tile_sample_min_size
        rows = []
        for i in range(0, z.shape[3], self.tile_latent_min_size):
            row = []
            for j in range(0, z.shape[4], self.tile_latent_min_size):
                tile = z[:, :, :, i : i + tilding_stride, j : j + tilding_stride]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, x0_feats[i // self.tile_latent_min_size][j // self.tile_latent_min_size])
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))
        return torch.cat(result_rows, dim=3)
        
    def decode(self, m, x0_feats, use_tiling=False):
        if not self.use_3d_conv:
            m = rearrange(m, "b c t h w -> (b t) c h w")
        if use_tiling:
            x = self.tiled_decode(m, x0_feats)
        else:
            m = self.post_quant_conv(m)
            x = self.decoder(m, x0_feats)

        time_padding = (
            0
            if self.time_downsample_factor ==0 or (self.num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - self.num_frames % self.time_downsample_factor
        )
        x = x[:, :, time_padding:]
        return x
    
    def forward_video(self, x, x_2d):
        use_tiling = self.use_tiling and x.shape[3] > int(self.tile_sample_min_size * (1 + self.tile_overlap_factor))
        
        if self.enable_fuse:
            if self.freeze_2d:
                with torch.no_grad():
                    x0_feats = self.encode_2d(x_2d, use_tiling)
            else:
                    x0_feats = self.encode_2d(x_2d, use_tiling)
            x0_feats = [x0_feats[i] if (i in self.ch_fuse or i == len(x0_feats) - 1) else None for i in range(len(x0_feats))]
        else:
            x0_feats = None   
        
        if self.freeze_3d :
            with torch.no_grad():
                posterior = self.encode_3d(x, use_tiling)
        else:
            posterior = self.encode_3d(x, use_tiling)
        
        motion = posterior.sample()
        motion = F.layer_norm(motion, motion.shape[1:])
            
        x_recon = self.decode(motion, x0_feats, use_tiling)
        return x_recon, posterior, motion
    
    def forward(self, batch):
        frame_length = torch.tensor(batch.shape[2], dtype=torch.long).to(self.device)
        center_index = (frame_length - 1) // 2
        start_frame = 0
        batch_input = batch[:, :, 0:]
        image_input = batch[:, :,center_index]
        return self.forward_video(batch_input, image_input)
            
        
    def get_input(self, batch):
        x = batch[self.image_key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 5:
            x = x.permute(0, 2, 1, 3, 4)
        x = x.to(memory_format=torch.contiguous_format) #.float()
        return x

    def training_step(self, batch, batch_idx): #optimizer_idx
        self.train()
        inputs = self.get_input(batch)
        start_time = time.time()
        
        reconstructions, posterior, motion = self(inputs)
        
        total_time = time.time() - start_time
        self.log("model_time",  total_time, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        start_time = time.time()
        
        optimizer, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer)
        
        start_frame = 0 
        
        if self.mode == 'residual': 
            center_index = (inputs.shape[2] - 1) // 2
            inputs = torch.cat([inputs[:, :, : center_index], inputs[:, :, center_index + 1:]], 2)
        
            
        aeloss, log_dict_ae = self.loss(inputs[:, :, start_frame:], reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(aeloss)
        

        if self.gradient_clip_val is not None:
            self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_time = time.time() - start_time
        self.log("back_g_time",  total_time, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        start_time = time.time()
        
        self.untoggle_optimizer(optimizer)
        self.toggle_optimizer(optimizer_d)

        discloss, log_dict_disc = self.loss(inputs[:, :, start_frame:], reconstructions, posterior, 1, self.global_step,last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        
        self.manual_backward(discloss)
        

        if self.gradient_clip_val_disc is not None:
            self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val_disc, gradient_clip_algorithm="norm")
        optimizer_d.step()     
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
    
        if self.use_scheduler:
            scheduler_g, scheduler_d = self.lr_schedulers()
            scheduler_g.step()
            scheduler_d.step()
            
        total_time = time.time() - start_time
        self.log("back_d_time",  total_time, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    
    @torch.no_grad()    
    def validation_step(self, batch, batch_idx):
        self.eval()
        start_frame =0 
        
        with torch.no_grad():
            x = self.get_input(batch)
            xrec, posterior, _ = self(x)
            x = rearrange(x[:, :, start_frame:], "b c t h w -> (b t) c h w").clamp(-1.0, 1.0)
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w").clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            xrec = (xrec + 1.0) / 2.0   # -1,1 -> 0,1; c,h,w
            psnr = compute_psnr(xrec, x)
            ssim = compute_ssim(xrec, x)
            lpips = self.loss.perceptual_loss(x, xrec).mean()

        self.log("psnr", psnr, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("ssim", ssim, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lpips", lpips.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        start_frame =0 
        with torch.no_grad():
            x = self.get_input(batch)
            xrec, posterior, _ = self(x)
            x = rearrange(x[:, :, start_frame:], "b c t h w -> (b t) c h w").clamp(-1.0, 1.0)
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w").clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            xrec = (xrec + 1.0) / 2.0   # -1,1 -> 0,1; c,h,w
            psnr = compute_psnr(xrec, x)
            ssim = compute_ssim(xrec, x)
            lpips = self.loss.perceptual_loss(x, xrec).mean()

        self.log("psnr", psnr, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("ssim", ssim, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lpips", lpips.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        lr_disc = self.disc_loss_scale *lr
        
        params = list(self.encoder_3d.parameters())+ list(self.decoder.parameters())+ \
                        list(self.quant_conv.parameters())+ list(self.post_quant_conv.parameters())
        
        if self.enable_2d:
            params += list(self.encoder_2d.parameters())
                
        if self.loss.logvar.requires_grad:
            params.append(self.loss.logvar)
        
        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        
        disc_params = []
        if self.loss.enable_2d:
            disc_params  +=  list(self.loss.discriminator_2d.parameters())
        if self.loss.enable_3d:
            disc_params  +=  list(self.loss.discriminator.parameters())
        
        opt_disc = torch.optim.Adam(disc_params, lr=lr_disc, betas=(0.5, 0.9))            

        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = {
                    "scheduler": LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,}
        else:
            scheduler = None
            
        if self.use_scheduler_disc:
            scheduler_disc = instantiate_from_config(self.scheduler_disc_config)
            print("Setting up LambdaLR scheduler...")
            scheduler_disc = {
                    "scheduler": LambdaLR(opt_disc, lr_lambda=scheduler_disc.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            scheduler = [scheduler, scheduler_disc]
        else:
            scheduler = []
            
        return [opt_ae, opt_disc], scheduler

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        
        log["inputs"] = rearrange(x, "b c t h w -> (b t) c h w")
        if not only_inputs:
            xrec, posterior, _ = self(x)
            log["reconstructions"] = rearrange(xrec, "b c t h w -> (b t) c h w")
            log["inputs_samples_hint-video"] = torch.cat([x, xrec], dim=3,)
        return log


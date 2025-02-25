import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

from .attenion import MEMORY_LAYOUT, get_cu_seqlens

# FLEX-ATTENTION ESSENTIALS.
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
create_block_mask = torch.compile(create_block_mask)
from diffusers.models.attention_processor import Attention
from typing import Optional
from functools import partial, lru_cache

attn_outputs_teacher = []
attn_outputs = []
BLOCK_MASK = None
HEIGHT = None
WIDTH = None

@lru_cache
def init_local_mask_flex(height, width, text_length, window_size, device):
    
    def local_mask(b, h, q_idx, kv_idx):
        q_y = (q_idx - text_length) // width
        q_x = (q_idx - text_length) % width
        kv_y = (kv_idx - text_length) // width
        kv_x = (kv_idx - text_length) % width
        return torch.logical_or(torch.logical_or(q_idx < text_length, kv_idx < text_length),
                                (q_y - kv_y) ** 2 + (q_x - kv_x) ** 2 < window_size ** 2)
    
    global BLOCK_MASK, HEIGHT, WIDTH
    BLOCK_MASK = create_block_mask(local_mask, B=None, H=None, device=device,
                                   Q_LEN=text_length + height * width, 
                                   KV_LEN=text_length + height * width, _compile=True)
    HEIGHT = height
    WIDTH = width

def block_flex_attention(
    q,
    k,
    v,
    mode="flex",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    proportional_attention=True,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "flex":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_func = partial(flex_attention, block_mask=attn_mask)
            attn_func = torch.compile(attn_func, dynamic=False)

        train_seq_len = q.size(1)
        head_dim = q.size(3)
        print("qkv shape: ", q.shape, k.shape, v.shape)
        if proportional_attention:
            attention_scale = math.sqrt(math.log(10 * k.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)
        x = attn_func(
            q, k, v, scale=attention_scale
        )
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

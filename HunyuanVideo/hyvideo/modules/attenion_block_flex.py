import math
import time

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

# FLEX-ATTENTION ESSENTIALS.
from torch.nn.attention.flex_attention import (
    create_block_mask, 
    flex_attention, 
    _convert_mask_to_block_mask, 
    _create_sparse_block_from_block_mask
)
create_block_mask = torch.compile(create_block_mask)
from diffusers.models.attention_processor import Attention
from typing import Optional
from functools import partial, lru_cache

attn_outputs_teacher = []
attn_outputs = []
BLOCK_MASK = None
HEIGHT = None
WIDTH = None

MEMORY_LAYOUT_BLOCK = {
    "flex": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


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

def tensor_to_block_mask(mask_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    # just use a full mask for now
    full_mask = torch.ones_like(mask_tensor, dtype=torch.bool)
    # TODO: merge the mask_mod function with the models_mul.py's repeat_interleave mode.
    BLOCK_SIZE_PER = 128
    # print("original mask percentage: ", mask_tensor.sum()/mask_tensor.numel())
    def mask_mod(b, h, q_idx, kv_idx):
        
        # print(q_idx, kv_idx)
        # 1. txt tokens can attend to all image tokens and text tokens
        kv_mask_idx = kv_idx // KV_BLOCK_SIZE
        q_mask_idx = q_idx // Q_BLOCK_SIZE
        return full_mask[q_mask_idx, kv_mask_idx]
    
        return torch.logical_or(
            q_idx >= 880*BLOCK_SIZE_PER,
            torch.logical_or(
                kv_idx >= 880*BLOCK_SIZE_PER,
                mask_tensor[q_mask_idx, kv_mask_idx]==True,
            )
        )
    
    block_mask = create_block_mask(mask_mod, B=None, H=None, device=mask_tensor.device,
                                   Q_LEN=Q_LEN, 
                                   KV_LEN=KV_LEN,
                                   BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
                                   _compile=True)
    # print sparsity
    # print("block_mask sparsity: ", block_mask.sparsity)
    return block_mask

def block_flex_attention(
    q,
    k,
    v,
    mode="flex",
    drop_rate=0,
    attn_mask=None,
    block_mask=None,
    causal=False,
    proportional_attention=False,
    KV_BLOCK_SIZE=128,
    Q_BLOCK_SIZE=128,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT_BLOCK[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "flex":
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(q.dtype)
            
            block_mask = tensor_to_block_mask(attn_mask, Q_LEN=q.size(2), KV_LEN=k.size(2), Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE)
            # calculate the compile time...
            t = time.time()
            attn_func = partial(flex_attention, block_mask=block_mask)
            attn_func = torch.compile(attn_func, dynamic=False)
            # print(f"compile time: {time.time() - t}")
            # print(block_mask)

        train_seq_len = q.size(2)
        head_dim = q.size(1)

        if proportional_attention:
            attention_scale = math.sqrt(math.log(10 * k.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)
        x = attn_func(
            q, k, v, scale=attention_scale
        )# .permute(0, 2, 1, 3)
        
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

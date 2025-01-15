import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

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

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}

# We set an global counter of timestep and layer
TIMESTEP = 0
LAYER_CUR = 0
TOTAL_LAYERS = 0

SAVE_PATH = "/dataset-vlm/yc/FinalProj/FlashVideo/results/results_attn_save"

from typing import Optional, Tuple
from hyvideo.modules.modulate_layers import modulate, apply_gate
from hyvideo.modules.posemb_layers import apply_rotary_emb
from hyvideo.modules.attenion import attention, parallel_attention
from einops import rearrange

def forward_double(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ):
    (
        img_mod1_shift,
        img_mod1_scale,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
    ) = self.img_mod(vec).chunk(6, dim=-1)
    (
        txt_mod1_shift,
        txt_mod1_scale,
        txt_mod1_gate,
        txt_mod2_shift,
        txt_mod2_scale,
        txt_mod2_gate,
    ) = self.txt_mod(vec).chunk(6, dim=-1)

    # Prepare image for attention.
    img_modulated = self.img_norm1(img)
    img_modulated = modulate(
        img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
    )
    img_qkv = self.img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(
        img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
    )
    # Apply QK-Norm if needed
    img_q = self.img_attn_q_norm(img_q).to(img_v)
    img_k = self.img_attn_k_norm(img_k).to(img_v)

    # Apply RoPE if needed.
    if freqs_cis is not None:
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

    # Prepare txt for attention.
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = modulate(
        txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
    )
    txt_qkv = self.txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(
        txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
    )
    # Apply QK-Norm if needed.
    txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
    txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

    # Run actual attention.
    q = torch.cat((img_q, txt_q), dim=1)
    k = torch.cat((img_k, txt_k), dim=1)
    v = torch.cat((img_v, txt_v), dim=1)
    assert (
        cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
    ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
    
    # just save the qk
    global TIMESTEP, LAYER_CUR, TOTAL_LAYERS
    if LAYER_CUR == 60:
        TIMESTEP += 1
        LAYER_CUR = 1
    else:
        LAYER_CUR += 1
        
    if TIMESTEP % 1 == 0:
        # save the qk
        torch.save(q.detach().cpu(), f"{SAVE_PATH}/q_double_{LAYER_CUR}_{TIMESTEP}.pt")
        torch.save(k.detach().cpu(), f"{SAVE_PATH}/k_double_{LAYER_CUR}_{TIMESTEP}.pt")
    
    # attention computation start
    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )
    else:
        attn = parallel_attention(
            self.hybrid_seq_parallel_attn,
            q,
            k,
            v,
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv
        )
        
    # attention computation end

    img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

    # Calculate the img bloks.
    img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
    img = img + apply_gate(
        self.img_mlp(
            modulate(
                self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
            )
        ),
        gate=img_mod2_gate,
    )

    # Calculate the txt bloks.
    txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
    txt = txt + apply_gate(
        self.txt_mlp(
            modulate(
                self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
            )
        ),
        gate=txt_mod2_gate,
    )

    return img, txt

def forward_single(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
    mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
    x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
    qkv, mlp = torch.split(
        self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
    )

    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

    # Apply QK-Norm if needed.
    q = self.q_norm(q).to(v)
    k = self.k_norm(k).to(v)

    # Apply RoPE if needed.
    if freqs_cis is not None:
        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)

    # Compute attention.
    assert (
        cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
    ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
    
    # just save the qk
    global TIMESTEP, LAYER_CUR, TOTAL_LAYERS
    if LAYER_CUR == 60:
        TIMESTEP += 1
        LAYER_CUR = 1
    else:
        LAYER_CUR += 1
    if TIMESTEP % 1 == 0:
        # save the qk
        torch.save(q.detach().cpu(), f"{SAVE_PATH}/q_single_{LAYER_CUR}_{TIMESTEP}.pt")
        torch.save(k.detach().cpu(), f"{SAVE_PATH}/k_single_{LAYER_CUR}_{TIMESTEP}.pt")
    
    # attention computation start
    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )
    else:
        attn = parallel_attention(
            self.hybrid_seq_parallel_attn,
            q,
            k,
            v,
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv
        )
    # attention computation end

    # Compute activation in mlp stream, cat again and run second linear layer.
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    return x + apply_gate(output, gate=mod_gate)


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # get numbner of layers
    print(hunyuan_video_sampler.model.double_blocks, hunyuan_video_sampler.model.single_blocks)
    
    # set total_layers to the number of layers
    global TOTAL_LAYERS
    TOTAL_LAYERS = hunyuan_video_sampler.model.double_blocks + hunyuan_video_sampler.model.single_blocks
    # uniformly replace the attention function with attn_with_save.
    for single_block in hunyuan_video_sampler.model.single_blocks:
        single_block.__class__.forward = forward_single
    for double_block in hunyuan_video_sampler.model.double_blocks:
        double_block.__class__.forward = forward_double
        
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()

import torch
import json
import numpy as np
import functools
import os
import time

from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens

from typing import Optional


import torch
import torch.distributed as dist

try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
    )
except:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None

import torch.profiler

def teacache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ):
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        
        if self.enable_teacache:
            inp = img.clone()
            vec_ = vec.clone()
            txt_ = txt.clone()
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
            normed_inp = self.double_blocks[0].img_norm1(inp)
            modulated_inp = modulate(
                normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
            )
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                local_rel_l1 = rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                
                # 同步所有GPU的rel_l1值
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    rel_l1_tensor = torch.tensor([local_rel_l1], device=modulated_inp.device)
                    dist.all_reduce(rel_l1_tensor, op=dist.ReduceOp.SUM)
                    avg_rel_l1 = rel_l1_tensor.item() / world_size
                    self.accumulated_rel_l1_distance += avg_rel_l1
                else:
                    self.accumulated_rel_l1_distance += local_rel_l1
                    
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                    
            # 确保所有GPU的should_calc一致
            if dist.is_initialized():
                should_calc_tensor = torch.tensor([1 if should_calc else 0], device=modulated_inp.device)
                dist.all_reduce(should_calc_tensor, op=dist.ReduceOp.MAX)
                should_calc = bool(should_calc_tensor.item())
                
            self.previous_modulated_input = modulated_inp  
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0
        
        if self.enable_teacache:
            # if dist.is_initialized():
            #     rank = dist.get_rank()
            #     logger.debug(f"Rank {rank} entering teacache_forward, step {self.cnt}, {should_calc}")
            if not should_calc:
                if dist.is_initialized():
                    rank = dist.get_rank()
                img += self.previous_residual
            else:
                ori_img = img.clone()
                # --------------------- Pass through DiT blocks ------------------------
                for _, block in enumerate(self.double_blocks):
                    double_block_args = [
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                    ]

                    img, txt = block(*double_block_args)

                # Merge txt and img to pass through single stream blocks.
                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for _, block in enumerate(self.single_blocks):
                        single_block_args = [
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                        ]

                        x = block(*single_block_args)

                img = x[:, :img_seq_len, ...]
                self.previous_residual = img - ori_img
        else:        
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]

                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

def parallelize_teacache_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.teacache_forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ):
        if x.shape[-2] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by height
            split_dim = -2
        elif x.shape[-1] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by width
            split_dim = -1
        else:
            raise ValueError(f"Cannot split video sequence into ulysses_degree x ring_degree ({get_sequence_parallel_world_size()}) parts evenly")

        # patch sizes for the temporal, height, and width dimensions are 1, 2, and 2.
        temporal_size, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2

        x = torch.chunk(x, get_sequence_parallel_world_size(), dim=split_dim)[get_sequence_parallel_rank()]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)

        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        
        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention()

        output = original_forward(
            x,
            t,
            text_states,
            text_mask,
            text_states_2,
            freqs_cos,
            freqs_sin,
            guidance,
            return_dict,
        )

        return_dict = not isinstance(output, tuple)
        sample = output["x"]
        sample = get_sp_group().all_gather(sample, dim=split_dim)
        output["x"] = sample
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

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
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = teacache_forward
    hunyuan_video_sampler.pipeline.transformer.__class__.teacache_forward = teacache_forward
    if hunyuan_video_sampler.parallel_args['ulysses_degree'] > 1 or hunyuan_video_sampler.parallel_args['ring_degree'] > 1:
        parallelize_teacache_transformer(hunyuan_video_sampler.pipeline)
        
    prompts = []
    # read from file.
    if args.prompt.endswith('.txt'):
        with open(args.prompt, 'r') as f:
            prompts = f.readlines()
    else:
        prompts = [args.prompt]
        
    for prompt in prompts:
        # TeaCache
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = True
        hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
        hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh = 0.15 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_modulated_input = None
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
        
        # Start sampling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{save_path}/profiler"),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            outputs = hunyuan_video_sampler.predict(
                prompt=prompt,
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=3,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale
            )
            prof.step()

        # Print profiler results
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        # Save detailed profiler results
        prof.export_chrome_trace(f"{save_path}/trace.json")

        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(outputs['samples']):
                sample = sample.unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                save_videos_grid(sample, save_path, fps=24)
                logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()

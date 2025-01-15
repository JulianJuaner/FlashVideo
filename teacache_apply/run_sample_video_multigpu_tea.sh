#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# Supported Parallel Configurations
# |     --video-size     | --video-length | --ulysses-degree x --ring-degree | --nproc_per_node |
# |----------------------|----------------|----------------------------------|------------------|
# | 1280 720 or 720 1280 | 129            | 8x1,4x2,2x4,1x8                  | 8                |
# | 1280 720 or 720 1280 | 129            | 1x5                              | 5                |
# | 1280 720 or 720 1280 | 129            | 4x1,2x2,1x4                      | 4                |
# | 1280 720 or 720 1280 | 129            | 3x1,1x3                          | 3                |
# | 1280 720 or 720 1280 | 129            | 2x1,1x2                          | 2                |
# | 1104 832 or 832 1104 | 129            | 4x1,2x2,1x4                      | 4                |
# | 1104 832 or 832 1104 | 129            | 3x1,1x3                          | 3                |
# | 1104 832 or 832 1104 | 129            | 2x1,1x2                          | 2                |
# | 960 960              | 129            | 6x1,3x2,2x3,1x6                  | 6                |
# | 960 960              | 129            | 4x1,2x2,1x4                      | 4                |
# | 960 960              | 129            | 3x1,1x3                          | 3                |
# | 960 960              | 129            | 1x2,2x1                          | 2                |
# | 960 544 or 544 960   | 129            | 6x1,3x2,2x3,1x6                  | 6                |
# | 960 544 or 544 960   | 129            | 4x1,2x2,1x4                      | 4                |
# | 960 544 or 544 960   | 129            | 3x1,1x3                          | 3                |
# | 960 544 or 544 960   | 129            | 1x2,2x1                          | 2                |
# | 832 624 or 624 832   | 129            | 4x1,2x2,1x4                      | 4                |
# | 624 832 or 624 832   | 129            | 3x1,1x3                          | 3                |
# | 832 624 or 624 832   | 129            | 2x1,1x2                          | 2                |
# | 720 720              | 129            | 1x5                              | 5                |
# | 720 720              | 129            | 3x1,1x3                          | 3                |

export TOKENIZERS_PARALLELISM=false

export NPROC_PER_NODE=8
export ULYSSES_DEGREE=8
export RING_DEGREE=1
export NCCL_P2P_DISABLE=0
# set NCCL timeout
export NCCL_TIMEOUT=1800
# add the current directory to the PYTHONPATH
export PYTHONPATH=/dataset-vlm/yc/FinalProj/FlashVideo/HunyuanVideo:$PYTHONPATH

# prompts=(
# 	"a woman in a red dress is dancing in a room",
# 	"A cat is sleeping on a sofa, in a living room.",
# 	"A cat walks on the grass, realistic style.",
# 	"A man is reading a book, in a library.",
# 	"A woman is cooking in a kitchen.",
# 	"A man is playing the guitar, in a studio.",
# 	"A woman is painting a picture, in a studio.",
# 	"A dog is playing with a ball, in a park.",
# 	"A cat is sleeping on a sofa, in a living room.",
# 	"A dog is playing with a frisbee, in a park.",
# 	"A cat is playing with a mouse, in a house.",
# 	"A man is playing the piano, in a concert hall."
# )
# use list in bash
prompts=("a woman in a red dress is dancing in a room" "A cat is sleeping on a sofa, in a living room." "A cat walks on the grass, realistic style." "A man is reading a book, in a library." "A woman is cooking in a kitchen." "A man is playing the guitar, in a studio." "A woman is painting a picture, in a studio." "A dog is playing with a ball, in a park." "A cat is sleeping on a sofa, in a living room." "A dog is playing with a frisbee, in a park." "A cat is playing with a mouse, in a house." "A man is playing the piano, in a concert hall.")
# loop over the prompt list.
for prompt in "${prompts[@]}"; do
	torchrun --nproc_per_node=$NPROC_PER_NODE /dataset-vlm/yc/FinalProj/FlashVideo/teacache_apply/teacache_sample_video.py \
		--video-size 720 1280 \
		--video-length 129 \
		--infer-steps 50 \
		--prompt "$prompt" \
		--seed 0 \
		--embedded-cfg-scale 6.0 \
		--flow-shift 7.0 \
		--flow-reverse \
		--ulysses-degree=$ULYSSES_DEGREE \
		--ring-degree=$RING_DEGREE \
		--save-path /dataset-vlm/yc/FinalProj/FlashVideo/results/results_tea_multigpu
done

# Exp Record.
# on node 31: 8-GPU inference: 398s
# on node 31: 1-GPU inference: 1390s

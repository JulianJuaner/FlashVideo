#!/bin/bash
export TOKENIZERS_PARALLELISM=false

export NPROC_PER_NODE=1
export ULYSSES_DEGREE=1
export RING_DEGREE=1
export NCCL_P2P_DISABLE=0
# set NCCL timeout
export NCCL_TIMEOUT=1800
# add the current directory to the PYTHONPATH
export PYTHONPATH=/dataset-vlm/yc/FinalProj/FlashVideo/HunyuanVideo:$PYTHONPATH

# use list in bash
prompts=("a woman in a red dress is dancing in a room")
# loop over the prompt list.
for prompt in "${prompts[@]}"; do
	python /dataset-vlm/yc/FinalProj/FlashVideo/attn_carve/sample_atten_save.py \
		--video-size 540 960 \
		--video-length 129 \
		--infer-steps 50 \
		--prompt "$prompt" \
		--seed 0 \
		--embedded-cfg-scale 6.0 \
		--flow-shift 7.0 \
    	--flow-reverse \
    	--use-cpu-offload \
		--ulysses-degree=$ULYSSES_DEGREE \
		--ring-degree=$RING_DEGREE \
		--save-path /dataset-vlm/yc/FinalProj/FlashVideo/results/results_attn_save
done

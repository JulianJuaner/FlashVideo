#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# enable debug
export CUDA_LAUNCH_BLOCKING=1

python3 sample_video.py \
    --video-size 704 1280 \
    --video-length 125 \
	--infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
	--embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results

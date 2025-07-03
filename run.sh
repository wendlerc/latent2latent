#!/bin/bash

# Small test script for paired dataset generation
echo "Starting small test of paired dataset generation..."

# Test with a small number of samples
python generate_paired_dataset.py \
    --data-url s3://cod-yt-latent-pairs/vids_pt/train \
    --output-path /home/developer/workspace/data/paired_test \
    --num-gpus 1 \
    --batch-size 8 \
    --sequence-length 5 \
    --shard-size-mb 10 \
    --dtype bfloat16 \
    --num-workers 1 \
    --max-samples 50 \
    --log-level DEBUG \

echo "Test completed!"
#!/bin/bash

# Small test script for paired dataset generation
echo "Starting small test of paired dataset generation..."

# Test with a small number of samples
python generate_paired_dataset_async.py \
    --data-url s3://cod-yt-latent-pairs/vids_pt/train \
    --output-path s3://cod-yt-latent-pairs/pairs/train2 \
    --num-gpus 1 \
    --sequence-length 101 \
    --shard-size-mb 100 \
    --dtype bfloat16 \
    --num-workers 1 \
    --log-level INFO \
    --start-from 1215

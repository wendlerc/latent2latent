python generate_paired_dataset.py generate-dataset \
    --data-root /mnt/nas/BlackOpsColdWar \
    --output-path /home/developer/workspace/data/paired/ \
    --num-gpus 1 \
    --sequence-length 9 \
    --shard-size-mb 10 \
    --dtype bfloat16 \
    --num-workers 2 \
    --max-samples 100
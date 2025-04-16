#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+8888)) \
--nproc_per_node=4 \
train.py \
--data-dir ./datasets/imagenet/ \
--batch-size 64 \
--model overlock_s \
--lr 1e-3 \
--auto-lr \
--drop-path 0.4 \
--epochs 300 \
--warmup-epochs 5 \
--workers 10 \
--model-ema \
--model-ema-decay 0.9999 \
--output output/overlock_s/ \
--native-amp \
--clip-grad 5
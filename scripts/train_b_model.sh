#!/usr/bin/env bash
torchrun \
--nproc_per_node=4 \
--master_port=$((RANDOM+8888)) \
train.py \
--data-dir ./datasets/imagenet/ \
--batch-size 128 \
--model overlock_b \
--lr 1e-3 \
--auto-lr \
--drop-path 0.5 \
--epochs 300 \
--warmup-epochs 5 \
--workers 10 \
--model-ema \
--model-ema-decay 0.9999 \
--output output/overlock_b/ \
--native-amp \
--clip-grad 5

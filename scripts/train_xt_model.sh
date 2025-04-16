#!/usr/bin/env bash
export OMP_NUM_THREADS=2
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+8888)) \
--nproc_per_node=4 \
train.py \
--data-dir ./datasets/imagenet/ \
--batch-size 160 \
--model overlock_xt \
--lr 1e-3 \
--auto-lr \
--drop-path 0.1 \
--epochs 300 \
--warmup-epochs 5 \
--workers 10 \
--model-ema \
--model-ema-decay 0.9999 \
--output output/overlock_xt/ \
--native-amp \
--clip-grad 5
#!/bin/bash

python -m scripts/inference_ETGP wandb=null experiment=hg38/enhancer_promoter_benchmark \
train.pretrained_model_path=dna_long_bench/ETGP/checkpoints/val/loss.ckpt \
train.test=True
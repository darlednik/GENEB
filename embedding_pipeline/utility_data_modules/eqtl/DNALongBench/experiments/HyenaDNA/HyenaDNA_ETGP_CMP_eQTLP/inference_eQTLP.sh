#!/bin/bash

python -m scripts/inference_eQTL wandb=null experiment=hg38/eqtl_benchmark \
train.pretrained_model_path=dna_long_bench/eQTL/checkpoints/val/loss.ckpt \
train.test=True
#!/bin/bash

python -m scripts/inference_contact_map wandb=null experiment=hg38/akita_benchmark \
train.pretrained_model_path=dna_long_bench/contact_map/checkpoints/val/loss.ckpt \
train.test=True
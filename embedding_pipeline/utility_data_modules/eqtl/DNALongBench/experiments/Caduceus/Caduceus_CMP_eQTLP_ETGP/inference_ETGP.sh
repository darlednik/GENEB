#!/bin/bash

python -m scripts/inference_eQTLP wandb=null \
experiment=hg38/enhancer_promoter_benchmark \
model=caduceus \
model._name_=caduceus_eqtl \
train.pretrained_model_path=caduceus/ph/ETGP/checkpoints/val/loss.ckpt \
train.test=True
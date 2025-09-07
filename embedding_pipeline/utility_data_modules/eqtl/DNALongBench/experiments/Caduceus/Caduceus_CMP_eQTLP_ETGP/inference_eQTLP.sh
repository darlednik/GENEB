#!/bin/bash

python -m scripts/inference_eQTLP wandb=null \
experiment=hg38/eqtl_benchmark \
model=caduceus \
model._name_=caduceus_eqtl \
train.pretrained_model_path=caduceus/ph/eQTLP/checkpoints/val/loss.ckpt \
train.test=True
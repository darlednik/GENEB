#!/bin/bash

python -m scripts/inference_contact_map wandb=null \
experiment=hg38/akita_benchmark \
model=caduceus \
model._name_=caduceus_contact_map \
train.pretrained_model_path=caduceus/ph/contact_map/checkpoints/val/loss.ckpt \
train.test=True
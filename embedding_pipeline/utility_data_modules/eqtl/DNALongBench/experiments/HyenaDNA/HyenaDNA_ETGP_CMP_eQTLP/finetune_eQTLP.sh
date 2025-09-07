#!/bin/bash

python -m scripts/finetune_eQTL wandb=null experiment=hg38/eqtl_benchmark \
train.pretrained_model_path=pretrained_models/weights.ckpt \
dataset.batch_size=1 \
hydra.run.dir=dna_long_bench/eQTL

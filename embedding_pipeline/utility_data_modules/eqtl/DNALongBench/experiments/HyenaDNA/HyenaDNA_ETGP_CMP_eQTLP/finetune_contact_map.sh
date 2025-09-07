#!/bin/bash


python -m scripts/finetune_contact_map wandb=null experiment=hg38/akita_benchmark \
train.pretrained_model_path=pretrained_models/weights.ckpt \
dataset.batch_size=1 \
hydra.run.dir=dna_long_bench/contact_map

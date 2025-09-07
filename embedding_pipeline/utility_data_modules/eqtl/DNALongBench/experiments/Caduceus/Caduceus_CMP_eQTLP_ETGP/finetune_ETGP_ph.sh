#!/bin/bash

python -m scripts/train_ETGP \
experiment=hg38/enhancer_promoter_benchmark \
hydra.run.dir=caduceus/ph/ETGP \
callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
dataset.dataset_name="enhancer_promoter" \
dataset.train_val_split_seed=1 \
dataset.batch_size=32 \
dataset.rc_aug=True \
+dataset.conjoin_train=false \
+dataset.conjoin_test=false \
model=caduceus \
model._name_=caduceus_finetune \
model.config.rcps=true \
trainer.max_epochs=30 \
wandb=null
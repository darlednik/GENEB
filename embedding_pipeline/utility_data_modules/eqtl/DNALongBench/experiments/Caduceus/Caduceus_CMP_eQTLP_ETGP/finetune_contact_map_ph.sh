#!/bin/bash


python -m scripts/train_contact_map \
experiment=hg38/akita_benchmark \
hydra.run.dir=caduceus/ph/contact_map \
callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
dataset.dataset_name="akita" \
dataset.train_val_split_seed=1 \
dataset.batch_size=32 \
dataset.rc_aug=true \
+dataset.conjoin_train=false \
+dataset.conjoin_test=false \
model=caduceus \
model._name_=caduceus_contact_map \
trainer.max_epochs=20 \
wandb=null
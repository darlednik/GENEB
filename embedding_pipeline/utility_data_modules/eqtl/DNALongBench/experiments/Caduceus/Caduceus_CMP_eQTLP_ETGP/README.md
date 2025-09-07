# Finetune Caduceus on CMP, eQTLP and ETGP tasks

## Install environment

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f caduceus_env.yml
```

Activate the environment.

```bash
conda activate caduceus_env
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir caduceus/ph
mkdir caduceus/ph/contact_map
mkdir caduceus/ph/eQTLP
mkdir caduceus/ph/ETGP
```

## Finetuning & Inference

### Contact Map Prediction Task

Taking caduceus-ph and cell type HFF as an example

Finetuning

```markdown
bash finetune_contact_map_ph
```

Inference

```markdown
bash inference_contact_map.sh
```

### eQTLP

Taking caduceus-ph and cell type CCF as an example

Finetuning

```markdown
bash finetune_eQTLP.sh
```

Inference

```markdown
bash finetune_eQTLP.sh
```

### ETGP

Taking caduceus-ph as an example

Finetuning

```markdown
bash finetune_ETGP.sh
```

Inference 

```markdown
bash inference_ETGP.sh
```
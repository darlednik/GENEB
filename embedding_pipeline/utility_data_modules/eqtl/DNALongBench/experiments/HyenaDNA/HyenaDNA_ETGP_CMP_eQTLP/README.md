# Finetune HyenaDNA on Contact Map Prediction, eQTLP and ETGP task

## Hugging Face pretrained weights
<a name="huggingface"></a>

- [medium-450k](https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen/tree/main)

## Dependencies
<a name="dependencies"></a>

For this repo, let's start with the dependancies that are needed. (If you're familiar with Docker, you can skip this section and jump to the [docker](#docker) setup below). The repo is built using Pytorch Lightning (a training library) and Hydra a config oriented ML library. (It'll be super helpful to get familiar with those tools.)

- clone repo, cd into it

```
git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git && cd hyena-dna
```

- create a conda environment, with Python 3.8+

```
conda create -n hyena-dna python=3.8
```

- The repo is developed with Pytorch 1.13, using cuda 11.7

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- install requirements:
```
pip install -r requirements.txt
```
- install Flash Attention, these [notes](https://github.com/HazyResearch/safari#getting-started) will be helpful.
```
cd hyena-dna
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e . --no-build-isolation
```
- optional fused layers for speed (takes a bit of time)
```
# from inside flash-attn/
cd csrc/layer_norm && pip install . --no-build-isolation
```

## Finetune Tasks

## Contact Map Prediction Task: Taking cell type HFF as an example

### Finetuning

```
finetune_contact_map.sh
```

### Inference
```markdown
bash inference_contact_map.sh
```

## eQTL Task

### Finetuning: Taking Cell type CCF as an example

```
finetune_eQTLP.sh
```

### Inference
```markdown
bash inference_eQTLP.sh
```

## ETGP Task

### Finetuning

```
finetune_ETGP.sh
```

### Inference
```markdown
bash inference_ETGP.sh
```



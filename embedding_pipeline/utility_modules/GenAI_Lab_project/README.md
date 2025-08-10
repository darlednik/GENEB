# GenAI-Lab

This repository is designated for the CSCI2952G group project.

# eccDNAMamba

A reproducible Python 3.9 environment using Conda and pip-based requirements.

## Environment Setup

### 1. Create the Conda environment with Python 3.9

```bash
conda create -n eccDNAMamba python=3.9
conda activate eccDNAMamba
```

### 2. Install PyTorch 2.1.0 with CUDA 11.8

```bash
pip install torch==2.2.0+cu121 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install nvcc via Conda

```bash
conda install -c "nvidia/label/cuda-12.1.105" cuda-nvcc
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt --no-deps
```
```bash
pip install datasets numpy==1.26.3 pyfaidx==0.8.1.3 mamba_ssm==2.2.4 causal_conv1d==1.5.0.post8 --no-cache-dir --no-build-isolation
```

### 5. Pre-training
```bash
python main.py
```

### 6. Fine-tuning

Download the **pretrained model weights** and **tokenizer files** from the following link, and place them into a folder named `weights` in your project directory:

[Pretrained Model Weights and Tokenizer](https://drive.google.com/drive/folders/1JUrzrE01Ud0Im7nSv0kJxpPlH6ub8GMd?usp=sharing)

```bash
python task1_finetune.py
```
```bash
python task2_finetune.py
```
```bash
python task3_finetune.py
```
### 7. Inference

```bash
python task1_inference.py
```
```bash
python task2_inference.py
```
```bash
python task3_inference.py
```

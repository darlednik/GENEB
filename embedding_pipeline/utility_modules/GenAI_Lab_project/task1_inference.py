import sys
sys.path.append("./GenAI-Lab")

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import csv
import torch
import math
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from safetensors.torch import load_file as load_safetensors
from BiMambaForMaskedLM import BiMambaForMaskedLM
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import Subset
import random
from pyfaidx import Fasta

# Load eccDNA data
hg19_genome = Fasta('data/genomes/hg19.fa')

# Load reference sequences
Healthy_person_input = "data/raw/CircleBase/Healthy_person.tsv"
Cancer_cell_line_input_input = "data/raw/CircleBase/Cancer_cell_line.tsv"

# Save sequences
os.makedirs("data/preprocessed/CircleBase", exist_ok=True)

# Function to extract sequences and write to a new file
def extract_sequences(input_file, genome, label=None):
    sequences = []

    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter='\t')

        for row in tqdm(reader, desc=f"Extracting {input_file}"):
            try:
                chrom = row['chr_hg19']  # Extract chromosome name
                start = int(float(row['start_hg19'])) - 1  # Convert start position to integer (0-indexed)
                end = int(float(row['end_hg19']))  # Convert end position to integer
                seq = genome[chrom][start:end].seq.upper()  # Extract sequence from genome file
                if 'N' in seq or end - start > 10000:  # Skip sequences with 'N' (unknown bases) and sequences too long
                    continue
            except Exception as e:
                continue

            sequences.append((seq.upper(), label))

    return sequences

# Extract sequences from datasets
Healthy_person_sequences  = extract_sequences(Healthy_person_input, hg19_genome, label=0)
Cancer_cell_line_sequences = extract_sequences(Cancer_cell_line_input_input, hg19_genome, label=1)

sequences = Healthy_person_sequences[:10000] + Cancer_cell_line_sequences[:10000]

print(f"Total sequences: {len(sequences)}")

# Load Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("saved_model_classifier_task1")

#Tokenize
train_data, eval_data = train_test_split(
    sequences,
    test_size=0.2,
    stratify=[l for _, l in sequences],
    random_state=42
)

def circular_augmentation(seq):
  head = tokenizer(seq, add_special_tokens=True, truncation=False)["input_ids"]
  circular_aug = head + head[1:65]

  return {"input_ids": circular_aug}

# Encoding function for non-causal language modeling
class EccDNADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenize):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        encoded = circular_augmentation(seq) 
        return {
            "input_ids": encoded["input_ids"],
            "labels": torch.tensor(label)
        }

train_dataset = EccDNADataset(train_data, tokenizer)
eval_dataset = EccDNADataset(eval_data, tokenizer)

# Model 
class BiMambaForClassification(PreTrainedModel):
    def __init__(self, backbone, hidden_size, config, num_classes=2, freeze_except_last_k=2):
        super().__init__(config)
        self.backbone = backbone

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last K layers of mamba_forward and mamba_backward
        if freeze_except_last_k > 0:
            for direction in [self.backbone.mamba_forward.backbone, self.backbone.mamba_backward.backbone]:
                layers = direction.layers  # nn.ModuleList
                for layer in layers[-freeze_except_last_k:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs["hidden_states"]  # (batch_size, seq_len, hidden_size)

        # Directly take the [CLS] token's hidden state
        cls_hidden = hidden[:, 0, :]  # (batch_size, hidden_size)

        logits = self.classifier(cls_hidden)  # (batch_size, num_classes)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

# Load backbone
def load_mamba(vocab_size):
    config = AutoConfig.from_pretrained("weights/", trust_remote_code=True)
    config.vocab_size = vocab_size
    config.pad_token_id = tokenizer.pad_token_id

    model = BiMambaForMaskedLM(config)
    return model, config

backbone, config = load_mamba(vocab_size=tokenizer.vocab_size)
state_dict = load_safetensors("weights/model.safetensors")
backbone.load_state_dict(state_dict)

backbone.eval()

model = BiMambaForClassification(backbone, hidden_size=768, config=config)

# Load classifier weights
state_dict = load_safetensors("saved_model_classifier_task1/model.safetensors")

# Load into the model
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

print("Done loading!")
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

model.eval()

# Dynamic Padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

# Eval Arguments
training_args = TrainingArguments(
    output_dir="./results/saved_model_classifier_task1_eval/",
    per_device_eval_batch_size=256,
    do_train=False,
    do_eval=True,
    bf16=True,
    report_to="none",
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    labels = torch.tensor(labels).numpy()
    f1 = f1_score(labels, preds, average='macro')
    return {"f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Eval 
NUM_EVALS = 10
eval_f1_scores = []

for i in range(NUM_EVALS):
    # Randomly select 80% of the evaluation set
    indices = random.sample(range(len(eval_dataset)), k=int(0.8 * len(eval_dataset)))
    subset = Subset(eval_dataset, indices)

    # Evaluate this subset with Trainer
    trainer.args.seed = random.randint(0, 10000)  # Change seed each time for true shuffling
    result = trainer.evaluate(eval_dataset=subset)
    eval_f1_scores.append(result["eval_f1"])

# Calculate mean and std of F1 scores
mean_f1 = np.mean(eval_f1_scores)
std_f1 = np.std(eval_f1_scores)

result = {
    "mean_f1": mean_f1,
    "std_f1": std_f1
}

print(f"Average F1 over 10 evaluations: {mean_f1:.4f}, Std Dev: {std_f1:.4f}")
print(f"All results: {eval_f1_scores}")

# Save evaluation results
import json
os.makedirs(training_args.output_dir, exist_ok=True) 
with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
    json.dump(result, f, indent=4)
print(f"Evaluation results saved to {training_args.output_dir}/eval_results.json")

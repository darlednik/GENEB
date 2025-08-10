from collections.abc import Sequence
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import pandas as pd
import itertools
from collections import Counter
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from BiMambaForMaskedLM import BiMambaForMaskedLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch
import math
import csv
from tqdm import tqdm
import sys
from itertools import islice
import random
from tokenizers.processors import TemplateProcessing

def tokenize_sequences():
  # Load and clean eccDNA sequences
  paths = [
      "data/preprocessed/CircleBase",
      "data/preprocessed/eccDNA_Atlas"
  ]
  
  os.makedirs("tmp/", exist_ok=True)

  target_files = {"Homo_sapiens.txt", "Gallus_gallus.txt", "Mus_musculus.txt"}
  seq_file = "tmp/sequences.txt"
  
  # Merge datasets
  sequence_set = set()

  with open(seq_file, "w", encoding="utf-8") as outfile:
      for path in paths:
          for filename in os.listdir(path):
              if filename in target_files:
                  file_path = os.path.join(path, filename)
                  with open(file_path, "r", encoding="utf-8") as infile:
                      for line in tqdm(infile, desc=f"Merging {filename} from {path}"):
                          line = line.strip()
                          if line not in sequence_set:
                              sequence_set.add(line)
                              outfile.write(line + "\n")

  sequences = list(sequence_set)
  print(f"Merge complete. Total sequences: {len(sequences)}. Saved to: {seq_file}")
  
  # Special tokens to match original setup
  special_tokens = ["[PAD]", "[CLS]", "[UNK]", "[MASK]"]
                   
  # Path to save the tokenizer
  tokenizer_path = "saved_model/"
  vocab_size = 4096
  
  # Initialize tokenizer and training configuration
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  tokenizer.pre_tokenizer = Whitespace()
  trainer = BpeTrainer(
      vocab_size=vocab_size,
      special_tokens=special_tokens
  )
  
  # Train from sequence
  tokenizer.train([seq_file], trainer)
  
  # Save trained tokenizer
  os.makedirs(tokenizer_path, exist_ok=True)
  tokenizer.save(f"{tokenizer_path}/tokenizer.json")
  
  # Wrap in PreTrainedTokenizerFast
  fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")
  fast_tokenizer.add_special_tokens({
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "unk_token": "[UNK]",
      "mask_token": "[MASK]",
  })


  fast_tokenizer._tokenizer.post_processor = TemplateProcessing(
      single="[CLS] $A",
      special_tokens=[
          ("[CLS]", fast_tokenizer.convert_tokens_to_ids("[CLS]")),
      ]
  )

  # Save all HuggingFace tokenizer files
  fast_tokenizer.save_pretrained(tokenizer_path)


  # Build Dataset and apply the tokenizer
  raw_dataset = Dataset.from_dict({"text": sequences})
  tokenized_dataset = raw_dataset.map(
      lambda batch: fast_tokenizer(batch["text"], add_special_tokens=True, truncation=False),
      batched=True,
      remove_columns=["text"]
  )
  
  tokenized_dataset.save_to_disk("tokenized_dataset/")
  
  total_tokens = sum(len(x["input_ids"]) for x in tokenized_dataset)
  print(f"Total tokens: {total_tokens}")

tokenize_sequences()

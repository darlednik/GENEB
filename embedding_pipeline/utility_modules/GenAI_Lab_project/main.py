import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from download_datasets import download_datasets
from extract_sequences import extract_sequences
from tokenize_sequences import tokenize_sequences
from train_model import train_model

# Download datasets
download_datasets()

# Extract sequences from datasets
extract_sequences()

# Tokenize datasets
tokenize_sequences()

# Train model
train_model()
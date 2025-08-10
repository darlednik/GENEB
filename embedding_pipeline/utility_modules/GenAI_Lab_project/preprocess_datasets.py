import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from download_datasets import download_datasets
from extract_sequences import extract_sequences
from tokenize_sequences import tokenize_sequences

# Download datasets
download_datasets()

# Extract sequences from datasets
extract_sequences()
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from tokenize_sequences import tokenize_sequences
from train_model import train_model

# Download datasets
tokenize_sequences()

# Extract sequences from datasets
train_model()
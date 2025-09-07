import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable, List, Any
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
)
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedModel,
    AutoConfig,
)
import dnalongbench
from dnalongbench.utils import load_data, BasenjiDataSet
import tensorflow as tf
from natsort import natsorted
import glob

def collate_fn(batch, tokenizer, max_length=450000):
    """
    Custom collate function for DNA data that converts one-hot encoded sequences to raw sequences
    and tokenizes them.
    
    Args:
        batch: List of tuples where each tuple is (x, y)
               x is one-hot encoded DNA sequence of shape (seq_len, 4)
               y is gene expression data of shape (10, seq_len)
        tokenizer: The GENERator tokenizer
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Dictionary with tokenized inputs and original gene expression data
    """
    # Separate x and y from the batch
    x_batch, y_batch = zip(*batch)

    # Convert one-hot encoded sequences to raw sequences
    raw_sequences = []
    nucleotides = ['A', 'C', 'G', 'T']
    for one_hot_seq in x_batch:
     
        # Ensure one_hot_seq is a PyTorch tensor
        if not isinstance(one_hot_seq, torch.Tensor):
            one_hot_seq = torch.tensor(one_hot_seq)
        
        # Get indices of 1s in one-hot encoding (argmax along axis 1)
        indices = torch.argmax(one_hot_seq, dim=1).cpu().numpy()
        
        # Convert indices to nucleotides
        raw_seq = ''.join([nucleotides[idx] for idx in indices])
        raw_sequences.append(raw_seq)
    
    # Tokenize the raw sequences
    tokenizer.padding_side = "right"
    inputs = tokenizer(
        raw_sequences,
        add_special_tokens=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        # max_length=max_length
    )

    
    # Convert y arrays to tensors and stack them
    y_tensors = []
    for y in y_batch:
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        y_tensors.append(y)
    
    y_stacked = torch.stack(y_tensors)
    
    # Return tokenized inputs and original y
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "y": y_stacked
    }

def collate_fn_eqtl(batch, tokenizer, max_length=450_000):
    """
    Ultra-fast version with further optimizations for very long sequences.
    """
    nucleotides = np.array(['A', 'C', 'G', 'T'], dtype='U1')  # Single character strings
    
    def one_hot_to_sequence(one_hot_array):
        """Ultra-fast conversion using numpy operations"""
        # Check for N positions more efficiently
        row_sums = np.sum(one_hot_array, axis=1)
        n_mask = np.abs(row_sums - 1.0) > 1e-6  # N positions sum to ~1.0, others sum to 1.0
        
        # Get argmax indices
        max_indices = np.argmax(one_hot_array, axis=1)
        
        # Create sequence array
        sequence_array = nucleotides[max_indices]
        
        # Set N positions
        if np.any(n_mask):
            sequence_array[n_mask] = 'N'
        
        # Fast join using numpy
        return sequence_array.tobytes().decode('ascii')
    
    # Process batch with minimal Python loops
    sequences_data = []
    y_values = []
    
    for item in batch:
        x_ref_seq = one_hot_to_sequence(item['x_ref'])
        x_alt_seq = one_hot_to_sequence(item['x_alt'])
        sequences_data.append((x_ref_seq, x_alt_seq))
        y_values.append(item['y'])
    
    # Separate sequences for tokenization
    x_ref_sequences, x_alt_sequences = zip(*sequences_data)
    
    # Tokenize in parallel if possible
    x_ref_tokenized = tokenizer(
        list(x_ref_sequences),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    x_alt_tokenized = tokenizer(
        list(x_alt_sequences),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    # Convert y values to tensor
    y_batch = torch.tensor([y.item() if hasattr(y, 'item') else y for y in y_values])
    
    return {
        'x_ref_input_ids': x_ref_tokenized["input_ids"],
        'x_ref_attention_mask': x_ref_tokenized["attention_mask"],
        'x_alt_input_ids': x_alt_tokenized["input_ids"],
        'x_alt_attention_mask': x_alt_tokenized["attention_mask"],
        'y': y_batch
    }

# Set logging level for transformers
transformers.logging.set_verbosity_info()

# Define optimization direction for each metric (whether higher or lower is better)
METRICS_DIRECTION: Dict[str, str] = {
    "accuracy": "max",
    "f1_score": "max",
    "mcc": "max",
    "f1_max": "max",
    "auprc_micro": "max",
    "mse": "min",
    "mae": "min",
    "r2": "max",
    "pearson": "max",
}


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0) in distributed training.

    Returns:
        bool: True if this is the main process, False otherwise
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)



def setup_tokenizer(
    model_name: str, padding_and_truncation_side: str
) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer for sequence understanding.

    Args:
        model_name: Name or path of the HuggingFace model
        padding_and_truncation_side: Side for padding and truncation (left or right)

    Returns:
        PreTrainedTokenizer: Configured tokenizer for the model
    """
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    start_time = time.time()

    # Load tokenizer with trust_remote_code to support custom models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Configure padding and truncation settings
    tokenizer.padding_side = padding_and_truncation_side
    tokenizer.truncation_side = padding_and_truncation_side

    # Set pad_token to eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dist_print(
        f"⏱️ Tokenizer loading completed in {time.time() - start_time:.2f} seconds"
    )

    return tokenizer




class AkitaDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfr_pattern, cell_type, target_lenth=99681):
        super(AkitaDataset).__init__()
        self.dataset = self.read_tfr(tfr_pattern)
        self.cell_type = cell_type
        target_ind_dict = {'HFF': 0, 'H1hESC': 1, 'GM12878': 2, 'IMR90': 3, 'HCT116': 4}
        self.target_ind = target_ind_dict[self.cell_type]
        self.target_length = target_lenth

    def file_to_records(self, filename):
        return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

    def parse_proto(self, example_protos):
        features = {
            'sequence': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string)
          }
        parsed_features = tf.io.parse_example(example_protos, features=features)
        seq = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
        targets = tf.io.decode_raw(parsed_features['target'], tf.float16)
        return seq, targets

    def read_tfr(self, tfr_pattern):
        tfr_files = natsorted(glob.glob(tfr_pattern))
        if tfr_files:
            dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
        else:
            print('Cannot order TFRecords %s' % tfr_pattern, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(tfr_pattern)
        dataset = dataset.flat_map(self.file_to_records)
        dataset = dataset.map(self.parse_proto)
        dataset = dataset.batch(1)
        return dataset

    def __iter__(self):
        num = 200
        for seq_raw, targets_raw in self.dataset:
            # print(seq_raw.shape, targets_raw.shape)\n",
            seq = seq_raw.cpu().numpy().reshape(-1, 4).astype('int8')
            targets = targets_raw.cpu().numpy().reshape(self.target_length, -1).astype('float16')
            # yield {"sequence": seq, "target": targets[:, self.target_ind]}
            # seq = seq[-372736: -65536, :]
            seq = seq[-475136: -65536, :]
            # seq = np.argmax(seq, axis=-1)
            targets = targets[:, self.target_ind]
            # targets = targets[-11026:]
            targets = targets[-19701:]
            scores = np.eye(num)
            index = 0
            for i in range(num):
                if i < num - 1:
                    scores[i][i + 1] = 1
                for j in range(i + 2, num):
                    scores[i][j] = targets[index]
                    index += 1
            for i in range(num):
                for j in range(i - 1):
                    scores[i][j] = scores[j][i]
            scores = torch.FloatTensor(scores).reshape(-1)
            yield (seq, scores)


def get_data(args):
    task_name = args.task_name
    if task_name == 'contact_map_prediction':
        tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices([], 'GPU')

        train_data_path = args.root + 'contact_map_prediction/targets/train-*.tfr'
        valid_data_path = args.root + 'contact_map_prediction/targets/valid-*.tfr'
        test_data_path = args.root + 'contact_map_prediction/targets/test-*.tfr'
        SEQUENCE_LENGTH = 1048576
        TARGET_LENGTH = 99681
        train_dataset = AkitaDataset(train_data_path, cell_type = args.subset)
        valid_dataset = AkitaDataset(valid_data_path, cell_type = args.subset)
        test_dataset = AkitaDataset(test_data_path, cell_type = args.subset)

        train_loader0 = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
        val_loader0 = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0)
        test_loader0 = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
        tokenizer = setup_tokenizer("GenerTeam/GENERator-eukaryote-1.2b-base", 'right')
        train_loader = DataLoader(
            train_loader0.dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length=409600)
        )
        val_loader = DataLoader(
            val_loader0.dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length=409600)
        )
        test_loader = DataLoader(
            test_loader0.dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length=409600)
        )

    elif task_name == 'regulatory_sequence_activity':
        train_loader0, valid_loader0, test_loader0 = load_data(root=args.root, task_name = args.task_name, subset = args.subset, batch_size=args.batch_size, sequence_length=196608)
        tokenizer = setup_tokenizer("GenerTeam/GENERator-eukaryote-1.2b-base", 'right')
        train_loader = DataLoader(
                train_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn(b, tokenizer)
            )

        val_loader = DataLoader(
            valid_loader0.dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )

        test_loader = DataLoader(
            test_loader0.dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer)
            )

    elif task_name == 'transcription_initiation_signal_prediction':
        train_loader0, valid_loader0, test_loader0 = load_data(root=args.root, task_name = args.task_name, batch_size=args.batch_size)
        tokenizer = setup_tokenizer("GenerTeam/GENERator-eukaryote-1.2b-base", 'right')
        max_samples = 10000 # take the first 10k examples

        train_loader = DataLoader(
                train_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn(b, tokenizer),
                shuffle=True
            )

        val_loader = DataLoader(
            Subset(valid_loader0.dataset, list(range(max_samples))),
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            shuffle=False
        )

        test_loader = None
    elif task_name == 'enhancer_target_gene_prediction':
        train_loader0, val_loader0, test_loader0 = load_data(root=args.root, task_name = args.task_name, batch_size=args.batch_size)
        tokenizer = setup_tokenizer("GenerTeam/GENERator-eukaryote-1.2b-base", 'right')

        train_loader = DataLoader(
                train_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn(b, tokenizer)
            )

        val_loader = DataLoader(
                val_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn(b, tokenizer)
            )

        test_loader = DataLoader(
                test_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn(b, tokenizer)
            )
    elif task_name == 'eqtl_prediction':
        train_loader0, val_loader0, test_loader0 = load_data(root=args.root, subset = args.subset, task_name = args.task_name, batch_size=args.batch_size)
        tokenizer = setup_tokenizer("GenerTeam/GENERator-eukaryote-1.2b-base", 'right')

        train_loader = DataLoader(
                train_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn_eqtl(b, tokenizer)
            )

        val_loader = DataLoader(
                val_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn_eqtl(b, tokenizer)
            )

        test_loader = DataLoader(
                test_loader0.dataset,
                batch_size=args.batch_size,
                collate_fn=lambda b: collate_fn_eqtl(b, tokenizer)
            )
    return train_loader, val_loader, test_loader
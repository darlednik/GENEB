"""
Data loader for NON-OVERLAPPING 6-mer tokenized sequences with lowercase tracking.
"""

import numpy as np
import tensorflow as tf
from typing import Any

# Feature description for reading TFRecords
def feature_description() -> Any:
    """Defines the structure of NON-OVERLAPPING 6-mer tokenized TFRecords."""
    return {
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "lowercase_mask": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "chromosome": tf.io.FixedLenFeature([], tf.string),
        "start_pos": tf.io.FixedLenFeature([], tf.int64),
        "end_pos": tf.io.FixedLenFeature([], tf.int64)
    }


def create_iterator(file_pattern: str, batch_size: int, shuffle: bool = True):
    """
    Creates an iterator for NON-OVERLAPPING 6-mer tokenized TFRecords.
    
    Args:
        file_pattern: Pattern for TFRecord files (e.g., "gs://bucket/path/record_*.tfrecord")
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
    
    Yields:
        Dictionary with keys: x, segment_ids, lowercase_mask, chromosome, start_pos, end_pos
    """
    
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description())
    
    # List all files matching the pattern
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    
    # Parse records
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Yield batches
    for batch in dataset:
        yield {
            "x": batch["x"].numpy().astype(np.int32),
            "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
            "lowercase_mask": batch["lowercase_mask"].numpy().astype(np.int32),
            "chromosome": batch["chromosome"].numpy(),
            "start_pos": batch["start_pos"].numpy().astype(np.int64),
            "end_pos": batch["end_pos"].numpy().astype(np.int64)
        }


def process_batch_6mer_nonoverlap(batch, cfg, step_idx: int | None = None, max_seq_len: int = 1536):
    """
    Process a batch for training with NON-OVERLAPPING 6-mer tokens and lowercase mask.
    
    Important: With non-overlapping 6-mer tokenization, sequences are 1366 tokens 
    (from 8192 nucleotides / 6 = 1365.33, rounded up to 1366).
    We pad to max_seq_len (1536) for flash attention compatibility.
    
    Args:
        batch: Dictionary from the iterator containing sequences and masks
        cfg: Config object with model parameters
        step_idx: Training step (unused but kept for compatibility)
        max_seq_len: Maximum sequence length to pad to (1536 for flash attention)
    
    Returns:
        Dictionary with x, y, segment_ids, and aux containing the lowercase mask
    """
    del step_idx  # Unused
    batch_size = batch["x"].shape[0]
    current_len = batch["x"].shape[1]  # Should be 1366
    
    # Pad sequences to max_seq_len (1536) for flash attention
    if current_len < max_seq_len:
        pad_len = max_seq_len - current_len
        # Pad with zeros (PADDING token is 0)
        batch["x"] = np.pad(batch["x"], ((0, 0), (0, pad_len)), constant_values=0)
        batch["segment_ids"] = np.pad(batch["segment_ids"], ((0, 0), (0, pad_len)), constant_values=0)
        batch["lowercase_mask"] = np.pad(batch["lowercase_mask"], ((0, 0), (0, pad_len)), constant_values=0)
    
    # For 6-mer, we still shift by 1 token (not 1 nucleotide)
    # Each token represents a 6-mer (6 nucleotides)
    patch_size = 1
    dummy = np.zeros((batch_size, patch_size), dtype=np.int32)
    
    # Prepare input and target sequences (shifted by patch_size for next-token prediction)
    x = np.concatenate([batch["x"][:, :-patch_size], dummy], axis=-1)
    y = np.concatenate([batch["x"][:, patch_size:], dummy], axis=-1)
    segment_ids = np.concatenate([batch["segment_ids"][:, :-patch_size], dummy], axis=-1)
    
    # Also shift the lowercase mask to align with targets
    lowercase_mask = np.concatenate([batch["lowercase_mask"][:, patch_size:], dummy], axis=-1)
    
    return {
        "x": x,
        "y": y,
        "segment_ids": segment_ids,
        "aux": {
            "lowercase_mask": lowercase_mask,  # This will be used for loss weighting
        }
    }

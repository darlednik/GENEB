"""
Data loader for 6-mer tokenized sequences with lowercase tracking.
"""

import numpy as np
import tensorflow as tf
from typing import Any

# Feature description for reading TFRecords
def feature_description() -> Any:
    """Defines the structure of 6-mer tokenized TFRecords."""
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
    Creates an iterator for 6-mer tokenized TFRecords.
    
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


def process_batch_6mer(batch, cfg, step_idx: int | None = None):
    """
    Process a batch for training with 6-mer tokens and lowercase mask.
    
    Important: With 6-mer tokenization, sequences are 8187 tokens (from 8192 nucleotides).
    We need to handle this carefully for next-token prediction.
    
    Args:
        batch: Dictionary from the iterator containing sequences and masks
        cfg: Config object with model parameters
        step_idx: Training step (unused but kept for compatibility)
    
    Returns:
        Dictionary with x, y, segment_ids, and aux containing the lowercase mask
    """
    del step_idx  # Unused
    batch_size = batch["x"].shape[0]
    
    # For 6-mer, we still shift by 1 token (not 1 nucleotide)
    # Each token represents a 6-mer
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

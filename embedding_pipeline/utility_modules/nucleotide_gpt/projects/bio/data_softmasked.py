"""
Data loader for softmasked human genome sequences with repetitive element tracking.

This module implements the data pipeline for training NucleotideGPT with repetitive
element (RE) weighting, as described in the paper "Probing Genomic Language Models: 
Nucleotide GPT and the Role of Pretraining in Learned Representations".

The softmasked genome format preserves RepeatMasker annotations where:
- Lowercase nucleotides = repetitive elements (will be downweighted during training)
- Uppercase nucleotides = unique sequences (full weight during training)

The TFRecords contain:
- Token sequences from the human genome (GRCh38)
- Binary masks indicating which positions are repetitive
- Segment IDs for handling padding
- Chromosome positions for reference

This data loader works in conjunction with the loss weighting in model.py to
implement the RE downweighting strategy that achieved optimal performance in
the paper (0.5 weighting for repetitive elements).

Usage:
    iterator = create_iterator("path/to/tfrecords/*.tfrecord", batch_size=16)
    for batch in iterator:
        processed = process_batch_softmasked(batch, cfg)
        # processed["aux"]["lowercase_mask"] contains the RE mask
"""

import numpy as np
import tensorflow as tf
from typing import Any

# Feature description for reading TFRecords
def feature_description() -> Any:
    """Defines the structure of softmasked TFRecords."""
    return {
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "lowercase_mask": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "chromosome": tf.io.FixedLenFeature([], tf.string),
        "start_pos": tf.io.FixedLenFeature([], tf.int64),
        "end_pos": tf.io.FixedLenFeature([], tf.int64)
    }


def create_iterator(file_pattern: str, batch_size: int, shuffle: bool = True):
    """Creates an iterator for softmasked TFRecords."""
    
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description())
    
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Match the original format - just return what we need
    for batch in dataset:
        yield {
            "x": batch["x"].numpy().astype(np.int32),
            "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
            "lowercase_mask": batch["lowercase_mask"].numpy().astype(np.int32),
            # Don't include chromosome, start_pos, end_pos
        }

def process_batch_softmasked(batch, cfg, step_idx: int | None = None):
    """Process batch - matching the style of process_batch in model.py"""
    del step_idx
    batch_size = batch["x"].shape[0]
    
    dummy = np.zeros((batch_size, 1), dtype=np.int32)
    
    # Extract lowercase_mask before processing
    lowercase_mask = batch.pop("lowercase_mask")  # Remove from batch
    
    # Process like normal
    x = np.concatenate([batch["x"][:, :-1], dummy], axis=-1)
    y = np.concatenate([batch["x"][:, 1:], dummy], axis=-1)
    segment_ids = np.concatenate([batch["segment_ids"][:, :-1], dummy], axis=-1)
    
    # Shift lowercase mask to align with y
    lowercase_mask_shifted = np.concatenate([lowercase_mask[:, 1:], dummy], axis=-1)
    
    return {
        "x": x,
        "y": y,
        "segment_ids": segment_ids,
        "aux": {
            "lowercase_mask": lowercase_mask_shifted
        }
    }

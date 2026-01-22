"""
Script to create TFRecords with NON-OVERLAPPING 6-mer tokenization for softmasked human genome.

Usage: python3 softmasked_human_6mer_nonoverlap.py --bucket-name minformer_data
"""

import argparse
import io
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from google.cloud import storage
from itertools import product

# Generate all possible 6-mers from ACGT
def generate_all_kmers(k=6, alphabet='ACGT'):
    """Generate all possible k-mers from the given alphabet."""
    return [''.join(p) for p in product(alphabet, repeat=k)]

# Create vocabulary
PADDING = "PADDING"
UNKNOWN = "UNKNOWN"  # For any 6-mer containing N or other non-ACGT characters

# Generate all 4^6 = 4096 possible 6-mers
all_6mers = generate_all_kmers(k=6, alphabet='ACGT')

# Create vocabulary with special tokens first
VOCAB = [PADDING, UNKNOWN] + all_6mers
VOCAB_SIZE = len(VOCAB)  # Should be 4098 (2 special tokens + 4096 6-mers)

# Create mapping dictionaries
stoi = {kmer: i for i, kmer in enumerate(VOCAB)}
itos = {i: kmer for i, kmer in enumerate(VOCAB)}

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"First few tokens: {VOCAB[:5]}")
print(f"Last few tokens: {VOCAB[-3:]}")

def tokenize_sequence_nonoverlapping(sequence, k=6):
    """
    Tokenize a DNA sequence using NON-overlapping k-mers.
    
    Args:
        sequence: DNA sequence string (can contain lowercase for softmasking)
        k: k-mer size (default 6)
    
    Returns:
        tokens: List of token IDs
        lowercase_mask: List of 1s (lowercase) and 0s (uppercase) matching token length
    """
    sequence = sequence.strip()
    tokens = []
    lowercase_mask = []
    
    # Process sequence in chunks of k (non-overlapping)
    for i in range(0, len(sequence), k):
        kmer = sequence[i:i+k]
        
        # Handle the last chunk if it's shorter than k
        if len(kmer) < k:
            # Pad with N's to make it a full k-mer
            kmer = kmer + 'N' * (k - len(kmer))
        
        # Check if this k-mer position is mostly lowercase (for softmasking)
        # We'll mark it as lowercase if more than half the characters are lowercase
        lowercase_count = sum(1 for c in kmer if c.islower())
        is_lowercase = 1 if lowercase_count > k//2 else 0
        lowercase_mask.append(is_lowercase)
        
        # Convert to uppercase for tokenization
        kmer_upper = kmer.upper()
        
        # If k-mer contains N or any non-ACGT character, map to UNKNOWN
        if any(c not in 'ACGT' for c in kmer_upper):
            tokens.append(stoi[UNKNOWN])
        else:
            tokens.append(stoi.get(kmer_upper, stoi[UNKNOWN]))
    
    return tokens, lowercase_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Create NON-overlapping 6-mer tokenized TFRecords for softmasked human genome")
    parser.add_argument("--bucket-name", type=str, required=True, help="GCS bucket name")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of sequences per TFRecord file")
    parser.add_argument("--max-tokens", type=int, default=1366, help="Maximum number of tokens per sequence (8192/6 = 1366)")
    return parser.parse_args()

def load_csv_from_gcs(bucket_name, file_path):
    """Load CSV from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_string()
    data_file = io.StringIO(data.decode("utf-8"))
    df = pd.read_csv(data_file)
    return df

def save_tfrecord(writer, tokens, segment_ids, lowercase_mask, chromosome, start_pos, end_pos):
    """
    Save a single example to TFRecord with 6-mer tokens and metadata.
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                # Tokenized sequence
                "x": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=tokens)
                ),
                # Segment IDs (1 for real sequence, 0 for padding)
                "segment_ids": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=segment_ids)
                ),
                # Lowercase mask (1 for softmasked regions, 0 for normal)
                "lowercase_mask": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=lowercase_mask)
                ),
                # Metadata
                "chromosome": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[chromosome.encode('utf-8')])
                ),
                "start_pos": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[start_pos])
                ),
                "end_pos": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[end_pos])
                )
            }
        )
    )
    writer.write(example.SerializeToString())

def process_dataframe(df, output_dir, batch_size=128, max_tokens=1366):
    """
    Process DataFrame and create TFRecord files with NON-overlapping 6-mer tokenization.
    """
    print(f"Processing {len(df)} sequences")
    print(f"Output directory: {output_dir}")
    print(f"Max tokens per sequence: {max_tokens} (non-overlapping 6-mers)")
    
    # Create GCS client for writing
    storage_client = storage.Client()
    bucket_name = output_dir.replace("gs://", "").split("/")[0]
    bucket = storage_client.bucket(bucket_name)
    base_path = "/".join(output_dir.replace(f"gs://{bucket_name}/", "").split("/"))
    
    record_count = 0
    batch_sequences = []
    
    # Check what columns we have
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Expected columns based on softmasked data
    sequence_col = 'sequence' if 'sequence' in df.columns else 'Sequence'
    chrom_col = 'chromosome' if 'chromosome' in df.columns else 'chrom' if 'chrom' in df.columns else 'Chrom'
    start_col = 'start' if 'start' in df.columns else 'start_pos' if 'start_pos' in df.columns else 'Start'
    end_col = 'end' if 'end' in df.columns else 'end_pos' if 'end_pos' in df.columns else 'End'
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        sequence = row[sequence_col]
        
        # Skip if sequence is too short
        if len(sequence) < 6:
            continue
            
        batch_sequences.append(row)
        
        # Write batch when we have enough sequences
        if len(batch_sequences) >= batch_size:
            output_file = f"{base_path}/record_{record_count}.tfrecord"
            
            # Create a temporary local file
            local_temp = f"/tmp/record_{record_count}.tfrecord"
            
            with tf.io.TFRecordWriter(local_temp) as writer:
                for batch_row in batch_sequences:
                    seq = batch_row[sequence_col]
                    
                    # Tokenize with NON-overlapping 6-mers
                    tokens, lowercase_mask = tokenize_sequence_nonoverlapping(seq, k=6)
                    
                    # Calculate padding needed
                    current_length = len(tokens)
                    padding_needed = max_tokens - current_length
                    
                    if padding_needed > 0:
                        # Pad tokens with PADDING token
                        tokens = tokens + [stoi[PADDING]] * padding_needed
                        # Create segment IDs (1 for real, 0 for padding)
                        segment_ids = [1] * current_length + [0] * padding_needed
                        # Pad lowercase mask with 0s
                        lowercase_mask = lowercase_mask + [0] * padding_needed
                    else:
                        # Truncate if too long
                        tokens = tokens[:max_tokens]
                        segment_ids = [1] * max_tokens
                        lowercase_mask = lowercase_mask[:max_tokens]
                    
                    # Get metadata with fallbacks
                    chromosome = str(batch_row.get(chrom_col, 'unknown'))
                    start_pos = int(batch_row.get(start_col, 0))
                    end_pos = int(batch_row.get(end_col, 0))
                    
                    # Save to TFRecord
                    save_tfrecord(
                        writer, 
                        tokens, 
                        segment_ids, 
                        lowercase_mask,
                        chromosome,
                        start_pos,
                        end_pos
                    )
            
            # Upload to GCS
            blob = bucket.blob(output_file)
            blob.upload_from_filename(local_temp)
            
            # Clean up local file
            os.remove(local_temp)
            
            print(f"Saved record_{record_count}.tfrecord with {len(batch_sequences)} sequences")
            record_count += 1
            batch_sequences = []
    
    # Handle remaining sequences
    if batch_sequences:
        output_file = f"{base_path}/record_{record_count}.tfrecord"
        local_temp = f"/tmp/record_{record_count}.tfrecord"
        
        with tf.io.TFRecordWriter(local_temp) as writer:
            for batch_row in batch_sequences:
                seq = batch_row[sequence_col]
                tokens, lowercase_mask = tokenize_sequence_nonoverlapping(seq, k=6)
                
                current_length = len(tokens)
                padding_needed = max_tokens - current_length
                
                if padding_needed > 0:
                    tokens = tokens + [stoi[PADDING]] * padding_needed
                    segment_ids = [1] * current_length + [0] * padding_needed
                    lowercase_mask = lowercase_mask + [0] * padding_needed
                else:
                    tokens = tokens[:max_tokens]
                    segment_ids = [1] * max_tokens
                    lowercase_mask = lowercase_mask[:max_tokens]
                
                chromosome = str(batch_row.get(chrom_col, 'unknown'))
                start_pos = int(batch_row.get(start_col, 0))
                end_pos = int(batch_row.get(end_col, 0))
                
                save_tfrecord(
                    writer, 
                    tokens, 
                    segment_ids, 
                    lowercase_mask,
                    chromosome,
                    start_pos,
                    end_pos
                )
        
        blob = bucket.blob(output_file)
        blob.upload_from_filename(local_temp)
        os.remove(local_temp)
        
        print(f"Saved final record_{record_count}.tfrecord with {len(batch_sequences)} sequences")

def main():
    args = parse_args()
    
    # Input and output paths
    input_path = "human_softmasked/hg38_softmasked_8192bp_bins.csv"
    output_dir = f"gs://{args.bucket_name}/human_softmasked_6mer_nonoverlap/tfrecords/"
    
    print(f"Loading softmasked human genome from gs://{args.bucket_name}/{input_path}")
    
    # Load the CSV
    df = load_csv_from_gcs(args.bucket_name, input_path)
    
    print(f"Loaded {len(df)} sequences")
    print(f"First few column names: {df.columns[:10].tolist()}")
    
    # Process and save as TFRecords
    process_dataframe(df, output_dir, args.batch_size, args.max_tokens)
    
    print("TFRecord creation complete!")
    print(f"Files saved to: {output_dir}")
    print(f"Sequence length: {args.max_tokens} tokens (from 8192 bp using non-overlapping 6-mers)")

if __name__ == "__main__":
    main()

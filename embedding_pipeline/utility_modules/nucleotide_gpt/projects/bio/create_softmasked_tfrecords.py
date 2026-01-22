"""
Script to convert softmasked human genome CSV to TFRecords with lowercase masking.
Usage: python3 create_softmasked_tfrecords.py
"""

import os
import io
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tqdm import tqdm
from google.cloud import storage

class SoftmaskedDNAProcessor:
    def __init__(self, sequence_length=8192):
        self.sequence_length = sequence_length

        # Vocabulary - treating uppercase and lowercase as same tokens
        self.PADDING = "P"
        self.VOCAB = [self.PADDING, "A", "C", "G", "T", "N"]
        self.VOCAB_SIZE = len(self.VOCAB)

        # Create mapping for both uppercase and lowercase
        self.stoi = {ch: i for i, ch in enumerate(self.VOCAB)}
        # Add lowercase mappings to same token IDs
        self.stoi.update({
            'a': self.stoi['A'],
            'c': self.stoi['C'],
            'g': self.stoi['G'],
            't': self.stoi['T'],
            'n': self.stoi['N']
        })

        self.itos = {i: ch for i, ch in enumerate(self.VOCAB)}

    def tokenize_with_mask(self, sequence):
        """
        Tokenize sequence and create lowercase mask.

        Returns:
            tokens: numpy array of token IDs
            lowercase_mask: numpy array where 1 = lowercase (repeat), 0 = uppercase (normal)
        """

        tokens = []
        lowercase_mask = []

        for char in sequence:
            # Get token ID (same for upper and lower)
            token_id = self.stoi.get(char.upper(), 0) # Default to padding if unknown
            tokens.append(token_id)

            # Track if original character was lowercase
            # 1 for lowercase, 0 for uppercase
            lowercase_mask.append(1 if char.islower() else 0)
        
        return np.array(tokens, dtype=np.int32), np.array(lowercase_mask, dtype=np.int32)
    
    def save_tfrecord(self, writer, tokens, segment_ids, lowercase_mask, chrom, start, end):
        """Save a single sequence to TFRecord with lowercase mask."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    # DNA sequence as token IDs
                    "x": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=tokens)
                    ),
                    # Segment IDs (1 for real sequence, 0 for padding)
                    "segment_ids": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=segment_ids)
                    ),
                    # Lowercase mask (1 for repeat/lowercase, 0 for normal/uppercase)
                    "lowercase_mask": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=lowercase_mask)
                    ),
                    # Genomic coordinates
                    "chromosome": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[chrom.encode()])
                    ),
                    "start_pos": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[start])
                    ),
                    "end_pos": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[end])
                    )
                }
            )
        )
        writer.write(example.SerializeToString())
    
    def process_csv_to_tfrecords(self, df, output_dir, sequences_per_file=128):
        """
        Process DataFrame and create TFRecord files.

        Args:
            df: DataFrame with columns [Chrom, Start, End, Sequence]
            output_dir: Output directory for TFRecord files
            sequences_per_file: Number of sequences to save per TFRecord file
        """
        print(f"Processing {len(df)} sequences to {output_dir}")

        # Shuffle the dataframe for better training
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        record_count = 0
        sequences_buffer = []

        # Track statistics
        total_lowercase = 0
        total_uppercase = 0
        skipped_sequences = 0

        for idx in tqdm(range(len(df)), desc="Processing sequences"):
            row = df.iloc[idx]
            sequence = row["Sequence"]

            # Skip sequences that are too short or have too many Ns
            if len(sequence) != self.sequence_length:
                skipped_sequences += 1
                print(f"Skipping sequence at index {idx}: wrong length {len(sequence)}")
                continue
            
            if sequence.upper().count('N') > 100: # Skip if too many unknown bases
                skipped_sequences += 1
                continue
            
            sequences_buffer.append(row)

            # Write to file when buffer is full
            if len(sequences_buffer) == sequences_per_file:
                output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")

                with tf.io.TFRecordWriter(output_file) as writer:
                    for seq_row in sequences_buffer:
                        # Tokenize with lowercase tracking
                        tokens, lowercase_mask = self.tokenize_with_mask(seq_row["Sequence"])

                        # No padding needed since sequences are already 8192bp
                        segment_ids = np.ones_like(tokens)

                        # Track statistics
                        total_lowercase += np.sum(lowercase_mask)
                        total_uppercase += np.sum(1 - lowercase_mask)

                        # Save to TFRecord
                        self.save_tfrecord(
                            writer,
                            tokens,
                            segment_ids,
                            lowercase_mask,
                            seq_row["Chrom"],
                            int(seq_row["Start"]),
                            int(seq_row["End"])
                        )
                
                record_count += 1
                sequences_buffer = []
        
        # Write remaining sequences
        if sequences_buffer:
            output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")
            with tf.io.TFRecordWriter(output_file) as writer:
                for seq_row in sequences_buffer:
                    tokens, lowercase_mask = self.tokenize_with_mask(seq_row["Sequence"])
                    segment_ids = np.ones_like(tokens)

                    total_lowercase += np.sum(lowercase_mask)
                    total_uppercase += np.sum(1 - lowercase_mask)

                    self.save_tfrecord(
                        writer,
                        tokens,
                        segment_ids,
                        lowercase_mask,
                        seq_row["Chrom"],
                        int(seq_row["Start"]),
                        int(seq_row["End"])
                    )
            record_count += 1
        
        # Print statistics
        print(f"\nProcessing complete!")
        print(f"Total TFRecord files created: {record_count}")
        print(f"Total sequences processed: {len(df) - skipped_sequences}")
        print(f"Sequences skipped: {skipped_sequences}")
        print(f"Lowercase (repeat) bases: {total_lowercase:,} ({100*total_lowercase/(total_lowercase+total_uppercase):.1f}%)")
        print(f"Uppercase (normal) bases: {total_uppercase:,} ({100*total_uppercase/(total_lowercase+total_uppercase):.1f}%)")

def load_csv_from_gcs(bucket_name, file_path):
    """Load CSV from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download as string and convert to DataFrame
    data = blob.download_as_string()
    data_file = io.StringIO(data.decode("utf-8"))
    df = pd.read_csv(data_file)

    return df

def main():
    # Configuration
    bucket_name = "minformer_data"
    csv_path = "human_softmasked/hg38_softmasked_8192bp_bins.csv"
    output_dir = f"gs://{bucket_name}/human_softmasked/tfrecords/"

    print(f"Loading CSV from gs://{bucket_name}/{csv_path}")

    # Load the CSV
    df = load_csv_from_gcs(bucket_name, csv_path)
    print(f"Loaded {len(df)} sequences")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize processor
    processor = SoftmaskedDNAProcessor(sequence_length=8192)

    # Process and create TFRecords
    processor.process_csv_to_tfrecords(df, output_dir)

    print(f"\nTFRecords saved to {output_dir}")

if __name__ == "__main__":
    main()

    

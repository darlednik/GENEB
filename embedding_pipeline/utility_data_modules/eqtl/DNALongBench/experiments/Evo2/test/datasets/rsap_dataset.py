import tensorflow as tf
import torch
import numpy as np
import os
import json
from natsort import natsorted
import glob
import sys
from torch.utils.data import Dataset, DataLoader, Subset
import functools
import pandas as pd
from kipoiseq.transforms.functional import one_hot_dna
from kipoiseq.dataclasses import Interval
from kipoiseq.extractors import FastaStringExtractor

tf.config.set_visible_devices([], "GPU")


task_name = 'regulatory_sequence_activity'
root = '/ocean/projects/bio240015p/shared/DNALongBench/'

batch_size = 1
one_hot = False


def get_dataloader(task_name = 'regulatory_sequence_activity', one_hot=False, subset = 'human'):
    assert subset in ('human', 'mouse'), "Subset must be 'human' or 'mouse'"
    human_fasta_path = root + 'regulatory_sequence_activity_prediction/human/seqs/hg38.ml.fa'
    mouse_fasta_path = root + 'regulatory_sequence_activity_prediction/mouse/seqs/mm10.ml.fa'
    data_path = root + 'regulatory_sequence_activity_prediction/'
    sequence_length = 196608
    fasta_path = human_fasta_path if subset == "human" else mouse_fasta_path

    train_dataset = BasenjiDataSet(data_path, subset, 'train', sequence_length, fasta_path, -1, one_hot)
    valid_dataset = BasenjiDataSet(data_path, subset, 'valid', sequence_length, fasta_path, -1, one_hot)
    test_dataset = BasenjiDataSet(data_path, subset, 'test', sequence_length, fasta_path, -1, one_hot)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    return train_loader, valid_loader, test_loader 


class BasenjiDataSet(torch.utils.data.IterableDataset):
    @staticmethod
    def get_organism_path(data_path, organism):
        return os.path.join(data_path, organism)

    @classmethod
    def get_metadata(cls, data_path, organism):
        # Keys:
        # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
        # pool_width, crop_bp, target_length
        path = os.path.join(cls.get_organism_path(data_path, organism), 'statistics.json')
        with tf.io.gfile.GFile(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def one_hot_encode(sequence):
        return kipoiseq.transforms.functional.one_hot_dna(sequence.upper()).astype(np.float32)

    @classmethod
    def get_tfrecord_files(cls, data_path, organism, subset):
        # Sort the values by int(*).
        return sorted(tf.io.gfile.glob(os.path.join(
            cls.get_organism_path(data_path, organism), 'targets', f'{subset}-*.tfr'
        )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    @property
    def num_channels(self):
        metadata = self.get_metadata(self.data_path, self.organism)
        return metadata['num_targets']

    @staticmethod
    def deserialize(serialized_example, metadata):
        """Deserialize bytes stored in TFRecordFile."""
        # Deserialization
        feature_map = {
            'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
            'target': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_example(serialized_example, feature_map)
        sequence = tf.io.decode_raw(example['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(example['target'], tf.float16)
        target = tf.reshape(target,
                            (metadata['target_length'], metadata['num_targets']))
        target = tf.cast(target, tf.float32)

        return {'sequence_old': sequence,
                'target': target}

    @classmethod
    def get_dataset(cls, data_path, organism, subset, num_threads=8):
        metadata = cls.get_metadata(data_path, organism)
        dataset = tf.data.TFRecordDataset(cls.get_tfrecord_files(data_path, organism, subset),
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_threads).map(
            functools.partial(cls.deserialize, metadata=metadata)
        )
        return dataset

    def __init__(self, data_path, organism: str, subset: str, seq_len: int, fasta_path: str, n_to_test: int = -1, one_hot: bool = True):
        assert subset in {"train", "valid", "test"}
        assert organism in {"human", "mouse"}
        self.data_path = data_path
        self.organism = organism
        self.subset = subset
        self.base_dir = self.get_organism_path(data_path, organism)
        self.seq_len = seq_len
        self.fasta_reader = FastaStringExtractor(fasta_path)
        self.n_to_test = n_to_test
        self.one_hot = one_hot 
        with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
            region_df = pd.read_csv(f, sep="\t", header=None)
            region_df.columns = ['chrom', 'start', 'end', 'subset']
            self.region_df = region_df.query('subset==@subset').reset_index(drop=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Only support single process loading"
        # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
        basenji_iterator = self.get_dataset(self.data_path, self.organism, self.subset, num_threads=1).as_numpy_iterator()
        for i, records in enumerate(basenji_iterator):
            loc_row = self.region_df.iloc[i]
            target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
            seq = self.fasta_reader.extract(target_interval.resize(self.seq_len))
            if self.one_hot:
                sequence_processed = self.one_hot_encode(seq)
            else:
                sequence_processed = seq 

            if self.n_to_test >= 0 and i < self.n_to_test and self.one_hot:
                old_sequence_onehot = records["sequence_old"]
                if old_sequence_onehot.shape[0] > sequence_processed.shape[0]:
                    diff = old_sequence_onehot.shape[0] - sequence_processed.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_processed)
                elif sequence_processed.shape[0] > old_sequence_onehot.shape[0]:
                    diff = sequence_processed.shape[0] - old_sequence_onehot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot, sequence_processed[trim:(-trim)])
                else:
                    np.testing.assert_equal(old_sequence_onehot, sequence_processed)

            yield sequence_processed, records["target"].copy()
   
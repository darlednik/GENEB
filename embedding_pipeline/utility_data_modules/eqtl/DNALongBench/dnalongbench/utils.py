#import tensorflow as tf
import torch
import sys
import os
import json
from collections import namedtuple
import logging
import random
#import selene_sdk
from abc import ABCMeta
#from selene_sdk.targets import Target, GenomicFeatures
#from selene_sdk.samplers import Sampler
#from selene_sdk.utils import get_indices_and_probabilities
# from selene_sdk.samplers import RandomPositionsSampler
#from selene_sdk.samplers.dataloader import SamplerDataLoader
import numpy as np
import pandas as pd
import pyfaidx
import functools
from kipoiseq import Interval
import os
from natsort import natsorted
import glob
from tqdm import tqdm
import tabix
import pyBigWig
import kipoiseq
from torch.utils.data import Dataset, DataLoader, Subset


def load_data(root='./DNALongBench/data', task_name = 'regulatory_sequence_activity', subset = 'human', batch_size=16, sequence_length=196608):
    assert task_name in ('regulatory_sequence_activity', 'contact_map_prediction', 'transcription_initiation_signal_prediction', 'enhancer_target_gene_prediction', 'eqtl_prediction'), "Choose a task from ('regulatory_sequence_activity', 'contact_map_prediction', 'transcription_initiation_signal_prediction', 'enhancer_target_gene_prediction', 'eqtl_prediction') "
    if task_name == 'regulatory_sequence_activity':
        assert subset in ('human', 'mouse'), "Subset must be 'human' or 'mouse'"
        human_fasta_path = root + 'regulatory_sequence_activity_prediction/human/seqs/hg38.ml.fa'
        mouse_fasta_path = root + 'regulatory_sequence_activity_prediction/mouse/seqs/mm10.ml.fa'
        data_path = root + 'regulatory_sequence_activity_prediction/'
        
        # SEQUENCE_LENGTH = 196608
        # BIN_SIZE = 128
        # TARGET_LENGTH = 896

        fasta_path = human_fasta_path if subset == "human" else mouse_fasta_path

        train_dataset = BasenjiDataSet(data_path, subset, 'train', sequence_length, fasta_path)
        valid_dataset = BasenjiDataSet(data_path, subset, 'valid', sequence_length, fasta_path)
        test_dataset = BasenjiDataSet(data_path, subset, 'test', sequence_length, fasta_path)

        def custom_collate(batch):
            x, y = zip(*batch)
            x = torch.tensor(np.stack(x), dtype=torch.float32)
            y = torch.tensor(np.stack(y), dtype=torch.float32)
            return x, y

        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        
   
        
    
    
    if task_name == 'enhancer_target_gene_prediction':
       
        ETGP_config_file = os.path.join(root, "enhancer_target_gene", "config", "CRISPRi_EPI_K562_hg19.config")
        task_root_path = os.path.join(root, "enhancer_target_gene")
        train_dataset = EPIseqDataSet(task_root_path, ETGP_config_file, 'train')
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size)
        valid_dataset = EPIseqDataSet(task_root_path, ETGP_config_file, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=batch_size)
        test_dataset = EPIseqDataSet(task_root_path, ETGP_config_file, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size)

    elif task_name == 'eqtl_prediction':
        
        supported_cell_types = ['Adipose_Subcutaneous', 'Artery_Tibial', 'Cells_Cultured_fibroblasts', 'Muscle_Skeletal', 'Nerve_Tibial', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Thyroid', 'Whole_Blood']
        # if cell_type not in supported_cell_type:
        #     print(f"Error: cell type {cell_type} is not supported", file = sys.stderr)
        #     print("Supported cell types are: ", supported_cell_type, file = sys.stderr)
        assert subset in supported_cell_types, (
            f"Error: cell type '{subset}' is not supported. "
            f"Supported cell types are: {supported_cell_types}"
        )
        eQTL_config_file = os.path.join(root, "eQTL", "config", "gtex_hg38.%s.config" % (subset))
        task_root_path = os.path.join(root, "eQTL")
        train_dataset = EQTLseqDataSet(task_root_path, eQTL_config_file, 'train')
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size)
        valid_dataset = EQTLseqDataSet(task_root_path, eQTL_config_file, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=batch_size)
        test_dataset = EQTLseqDataSet(task_root_path, eQTL_config_file, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size)
    
        
    return train_loader, valid_loader, test_loader





class AkitaDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfr_pattern, cell_type, target_length=99681):
        super(AkitaDataset).__init__()
        self.dataset = self.read_tfr(tfr_pattern)
        self.cell_type = cell_type
        target_ind_dict = {'HFF': 0, 'H1hESC': 1, 'GM12878': 2, 'IMR90': 3, 'HCT116': 4}
        self.target_ind = target_ind_dict[self.cell_type]
        self.target_length=target_length

    
    # Adapted from https://github.com/calico/basenji/blob/498d0b1cd02e1ba11658d273ef257a07f5a69657/bin/tfr_qc.py#L98
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
        for seq_raw, targets_raw in self.dataset:
            seq = seq_raw.numpy().reshape(-1,4).astype('int8')
            targets = targets_raw.numpy().reshape(self.target_length,-1).astype('float16')
            # yield {
            #       "sequence": seq,
            #       "target": targets[:, self.target_ind],
                
            #   }
            yield seq, targets[:, self.target_ind]


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


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
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

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

    def __init__(self, data_path, organism: str, subset: str, seq_len: int, fasta_path: str, n_to_test: int = -1):
        assert subset in {"train", "valid", "test"}
        assert organism in {"human", "mouse"}
        self.data_path = data_path
        self.organism = organism
        self.subset = subset
        self.base_dir = self.get_organism_path(data_path, organism)
        self.seq_len = seq_len
        self.fasta_reader = FastaStringExtractor(fasta_path)
        self.n_to_test = n_to_test
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
            sequence_one_hot = self.one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
            if self.n_to_test >= 0 and i < self.n_to_test:
                old_sequence_onehot = records["sequence_old"]
                if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
                    diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
                elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
                    diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
                else:
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)
            # yield {
            #     "sequence": sequence_one_hot,
            #     "target": records["target"],
            # }
            yield sequence_one_hot.copy(), records["target"].copy()




class EPIseqDataSet(torch.utils.data.IterableDataset):

    def __init__(self, root_path, config_file, subset):
        super(EPIseqDataSet).__init__()
        #########################
        # load config
        #########################
        self.config = parse_config(config_file)
        # init
        self.organism = 'Unknown'
        if self.config.get('organism', None) is not None:
            self.organism = self.config['organism']
        self.subset = subset
        if self.subset is None: 
            if self.config.get('subset', None) is not None:
                self.subset = self.config['subset']
            else:
                print("subset is required either in the config file or passed as a parameter", file = sys.stderr)
        try:
            assert self.subset in ['train', 'valid', 'test']
        except AssertionError:
            print("subset must be either train, valid, or test", file = sys.stderr)
        # verbose level
        self.verbose = 0
        if self.config.get('verbose', None) is not None:
            self.verbose = self.config['verbose']
        # show config
        if self.verbose > 0:
            print_config(self.config)
        print("> load config done")

        ############################
        # load fasta file
        ############################
        if self.config.get('genome_fa', None) is None:
            print('genome_fa is required in the config file', file = sys.stderr)
        self.fasta_reader = FastaStringExtractor(os.path.join(root_path, self.config['genome_fa']))
        print("> init fasta extractor done")

        ############################
        # load dataset, ie EPI data
        ############################
        if self.config.get('seq_len_cutoff', None) is None:
            print('seq_len_cutoff is required in the config file', file = sys.stderr)
            print('will use default value 450000', file = sys.stderr)
            self.config['seq_len_cutoff'] = 450000
        if self.config.get('enhancer_tabix_file', None) is None:
            print('enhancer_tabix_file is required in the config file', file = sys.stderr)
        self.dataset = parse_EPI(root_path, self.config['EPI_file'], self.config['enhancer_tabix_file'], self.fasta_reader, self.subset, self.config['seq_len_cutoff'], self.config['tss_flank_upstream'], self.config['tss_flank_downstream'], self.config['region_flank_upstream'], self.config['region_flank_downstream'])
        if self.verbose > 0:
            print(f'Loaded {len(self.dataset)} EPI records')

    def __iter__(self):
        # iterate over the data
        target2int = {'positive':1, 'negative':0}
        for index, record in enumerate(self.dataset):
            # convert the sequence to one-hot encoding
            sequence = one_hot_encode(record[0])
            target = np.array(target2int[record[1]]).astype('long')
            yield {
                'sequence': record[0],
                'label': target 
            }
            #yield sequence, target


def parse_EPI(root_path, EPI_file, blacklist_tabix_file, fasta_reader, subset_group_name, seq_len_cutoff = 450000, tss_flank_upstream = 3000, tss_flank_downstream = 3000, region_flank_upstream = 500, region_flank_downstream = 500):
    """
    this function will parse the following file format to reresent an enhancer-promoter interaction entry
        > gene_chrom  gene_start gene_end region_chrom region_start region_end gene_id gene_strand region_id target subset
        > other columns may exists but will be ignored
    then, the function will search for any enhancers between the tested region and the gene tss the sequence in these enhancers will be masked N. This is crucial because we want to ensure that these sequences don’t influence the model training process. 
    """
    # check subset
    if subset_group_name not in ['train', 'valid', 'test']:
        print(f'Error: unknown subset group {subset_group_name}')
    # load records as dataframe
    df = pd.read_csv(os.path.join(root_path, EPI_file), sep='\t', header=0)
    # load the tabix
    blacklist_tabix = tabix.open(os.path.join(root_path, blacklist_tabix_file))
    # parse the records and convert it into a list of DNA sequence and target (ie, true or false EPI)
    dataset = []
    print("> Start parsing EPI records to build the dataset %s" % (subset_group_name))
    N_total = df.shape[0]
    N_pass = 0
    N_skip_for_chrom_diff = 0 # skip if enhancer and gene tss are in different chromosomes
    N_skip_for_len = 0 # skip if the distance between enhancer and gene tss is too long
    N_skip_for_strand = 0 # skip if gene strand is not + or - or unknown
    N_skip_for_short = 0 # skip if the distance between enhancer and gene tss is 0, ie, they are overlapping
    for index, row in tqdm(df.iterrows(), total = N_total):
        gene_chrom = row['gene_chrom']
        gene_start = row['gene_start']
        gene_end = row['gene_end']
        region_chrom = row['region_chrom']
        region_start = row['region_start']
        region_end = row['region_end']
        gene_id = row['gene_id']
        gene_strand = row['gene_strand']
        region_id = row['region_id']
        target = row['target']
        subset_group = row['subset']
        # skip if in different subset group
        if subset_group != subset_group_name:
            continue
        # skip if in different chromosomes
        if gene_chrom != region_chrom:
            N_skip_for_chrom_diff += 1
            continue
        # get tss
        if gene_strand == '+':
            tss_start = gene_start
        elif gene_strand == '-':
            tss_start = gene_end - 1
        else:
            print(f'Warning: unknown strand {gene_strand}')
            N_skip_for_strand += 1
            continue
        tss_end = tss_start + 1
        # add flanking sequence
        tss_start = tss_start - tss_flank_upstream
        tss_end = tss_end + tss_flank_downstream
        region_start = region_start - region_flank_upstream
        region_end = region_end + region_flank_downstream
        # calculate the distance between tss and region
        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        # check if the sequence length is too long
        if distance > seq_len_cutoff:
            N_skip_for_len += 1
            continue
        # get the interverl between the region and tss
        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        # get the sequence
        region_interval = Interval(region_chrom, sequence_start, sequence_end)
        region_seq = fasta_reader.extract(region_interval)
        # mask any enhancer sequence
        if distance > 0:
            if tss_start > region_end:
                query_start = region_end
                query_end = tss_start
            else:
                query_start = tss_end
                query_end = region_start
            query_result = blacklist_tabix.query(region_chrom, query_start, query_end)
            for overlap in query_result:
                o_start = int(overlap[1])
                o_end = int(overlap[2])
                # mask the correspoinding DNA sequence as 'N'
                rev_start = o_start - sequence_start
                rev_end = o_end - sequence_start
                if rev_start > 0 and rev_start < len(region_seq) and rev_end > 0 and rev_end < len(region_seq):
                    region_seq = region_seq[:rev_start] + 'N' * (rev_end - rev_start) + region_seq[rev_end:]
        # debug
        #region_seq =  region_seq[:20000] + region_seq[-20000:]
        # flip (reverse complement) the sequence if gene is on the downstream of region (ie, genomic coordinate is larger)
        if gene_start > region_end:
            region_seq = rcDNA(region_seq) 
        # padding N to the sequence if the sequence
        # debug
        #if len(region_seq) <= 40000:
        #    region_seq = region_seq + 'N' * (40000 - len(region_seq))
        #else:
        #    region_seq = region_seq[:40000]
        if len(region_seq) <= seq_len_cutoff:
            region_seq = region_seq + 'N' * (seq_len_cutoff - len(region_seq))
        else:
            region_seq = region_seq[:seq_len_cutoff]
        N_pass +=1
        # append to the dataset
        dataset.append((region_seq, target))
    # report the statistics
    print("# Finish parsing EPI records")
    print("# Total records: ", N_total)
    print("# Skipped records due to different chromosomes: ", N_skip_for_chrom_diff)
    print("# Skipped records due to distance cutoff: ", N_skip_for_len)
    print("# Skipped records due to unknown strand: ", N_skip_for_strand)
    print("# Select records %s with subset %s " % (N_pass, subset_group_name))
    return dataset


def rcDNA(seq):
    """Get reverse complemented of a DNA sequence"""
    diccomseq = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
    return "".join(diccomseq[nu.upper()] for nu in seq[::-1])


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def parse_config(config_file):
    """
    load config as a dictionary
    """
    config_table = {}
    with open(config_file, 'r') as fin:
        for line in fin:
            row = line.strip().split()
            key = row[0]
            value = row[1]
            data_type = row[2]
            if data_type == 'int':
                value = int(value)
            elif data_type == 'float':
                value = float(value)
            elif data_type == 'bool':
                value = bool(value)
            elif data_type == 'str':
                value = str(value)
            elif data_type == 'list':
                value_list = value.split(',')
                # get the data type of the list
                new_value_list = []
                for value in value_list:
                    if value.isnumeric():
                        value = int(value)
                    elif value.replace('.','',1).isdigit():
                        value = float(value)
                    else:
                        value = str(value)
                    new_value_list.append(value)
                value = new_value_list
            else:
                value = str(value)
            #
            config_table[key] = value
    return config_table


def print_config(config_table):
    """
    print config
    """
    for key, value in config_table.items():
        print(f'{key}: {value}')


class EQTLseqDataSet(torch.utils.data.IterableDataset):

    def __init__(self, root_path, config_file, subset):
        super(EQTLseqDataSet).__init__()
        #########################
        # load config
        #########################
        self.config = parse_config(config_file)
        # init
        self.organism = 'Unknown'
        if self.config.get('organism', None) is not None:
            self.organism = self.config['organism']
        self.subset = subset
        if self.subset is None: 
            if self.config.get('subset', None) is not None:
                self.subset = self.config['subset']
            else:
                print("subset is required either in the config file or passed as a parameter", file = sys.stderr)
        try:
            assert self.subset in ['train', 'valid', 'test']
        except AssertionError:
            print("subset must be either train, valid, or test", file = sys.stderr)
        # verbose level
        self.verbose = 0
        if self.config.get('verbose', None) is not None:
            self.verbose = self.config['verbose']
        # show config
        if self.verbose > 0:
            print_config(self.config)
        print("> load config done")

        ############################
        # load fasta file
        ############################
        if self.config.get('genome_fa', None) is None:
            print('genome_fa is required in the config file', file = sys.stderr)
        self.fasta_reader = FastaStringExtractor(os.path.join(root_path, self.config['genome_fa']))
        print("> init fasta extractor done")

        ############################
        # load dataset, ie eQTL data
        ############################
        if self.config.get('seq_len_cutoff', None) is None:
            print('seq_len_cutoff is required in the config file', file = sys.stderr)
            print('will use default value 450000', file = sys.stderr)
            self.config['seq_len_cutoff'] = 450000
        if self.config.get('eQTL_tabix_file', None) is None:
            print('eQTL_tabix_file is required in the config file', file = sys.stderr)
        self.dataset = parse_eQTL(root_path, self.config['eQTL_file'], self.config['eQTL_tabix_file'], self.fasta_reader, self.subset, self.config['seq_len_cutoff'], self.config['tss_flank_upstream'], self.config['tss_flank_downstream'], self.config['region_flank_upstream'], self.config['region_flank_downstream'])
        if self.verbose > 0:
            print(f'Loaded {len(self.dataset)} eQTL records')

    def __iter__(self):
        # iterate over the data
        target2int = {'positive':1, 'negative':0}
        for index, record in enumerate(self.dataset):
            # convert the sequence to one-hot encoding
            sequence_ref = one_hot_encode(record[0])
            sequence_alt = one_hot_encode(record[1])
            sequence_ref_alt = np.concatenate([sequence_ref, sequence_alt], axis = 0)
            target = np.array(target2int[record[2]]).astype('long')
            yield {
                'x_ref': record[0],#sequence_ref,
                'x_alt': record[1],#sequence_alt,
                'y': target 
            }


def parse_eQTL(root_path, eQTL_file, blacklist_tabix_file, fasta_reader, subset_group_name, seq_len_cutoff = 450000, tss_flank_upstream = 3000, tss_flank_downstream = 3000, region_flank_upstream = 500, region_flank_downstream = 500):
    """
    this function will parse the following file format to reresent an variant-gene interaction entry
        > gene_chrom  gene_start gene_end region_chrom region_start region_end gene_id gene_strand region_id target subset allele1 allele2
        > other columns may exists but will be ignored
    then, the function will search for any variants between the tested region and the gene tss the sequence in these variants will be masked N. This is crucial because we want to ensure that these sequences don’t influence the model training process. 
    """
    # check subset
    if subset_group_name not in ['train', 'valid', 'test']:
        print(f'Error: unknown subset group {subset_group_name}')
    # load records as dataframe
    df = pd.read_csv(os.path.join(root_path, eQTL_file), sep='\t', header=0)
    # load the tabix
    blacklist_tabix = tabix.open(os.path.join(root_path, blacklist_tabix_file))
    # parse the records and convert it into a list of DNA sequence and target (ie, true or false eQTL)
    dataset = []
    print("> Start parsing eQTL records to build the dataset %s" % (subset_group_name))
    N_total = df.shape[0]
    N_pass = 0
    N_skip_for_chrom_diff = 0 # skip if variant and gene tss are in different chromosomes
    N_skip_for_len = 0 # skip if the distance between variant and gene tss is too long
    N_skip_for_strand = 0 # skip if gene strand is not + or - or unknown
    for index, row in tqdm(df.iterrows(), total = N_total):
        gene_chrom = row['gene_chrom']
        gene_start = row['gene_start']
        gene_end = row['gene_end']
        region_chrom = row['region_chrom']
        region_start = row['region_start']
        region_end = row['region_end']
        gene_id = row['gene_id']
        gene_strand = row['gene_strand']
        region_id = row['region_id']
        target = row['target']
        subset_group = row['subset']
        allele1 = row['allele1']
        allele2 = row['allele2']
        # skip if in different subset group
        if subset_group != subset_group_name:
            continue
        # skip if in different chromosomes
        if gene_chrom != region_chrom:
            N_skip_for_chrom_diff += 1
            continue
        # get tss
        if gene_strand == '+':
            tss_start = gene_start
        elif gene_strand == '-':
            tss_start = gene_end - 1
        else:
            print(f'Warning: unknown strand {gene_strand}')
            N_skip_for_strand += 1
            continue
        tss_end = tss_start + 1
        # add flanking sequence
        tss_start = tss_start - tss_flank_upstream
        tss_end = tss_end + tss_flank_downstream
        variant_start = region_start
        variant_end = region_end
        region_start = region_start - region_flank_upstream
        region_end = region_end + region_flank_downstream
        # calculate the distance between tss and region
        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        # check if the sequence length is too long
        if distance > seq_len_cutoff:
            N_skip_for_len += 1
            continue
        # get the interverl between the region and tss
        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        # get the sequence
        region_interval = Interval(region_chrom, sequence_start, sequence_end)
        region_seq = fasta_reader.extract(region_interval)
        # check if the variant sequence matches the reference sequence
        variant_rev_start = variant_start - sequence_start
        variant_rev_end = variant_end - sequence_start
        assert region_seq[variant_rev_start:variant_rev_end] == row['allele1']
        # mask any eQTL sequence
        if distance > 0:
            if tss_start > region_end:
                query_start = region_end
                query_end = tss_start
            else:
                query_start = tss_end
                query_end = region_start
            query_result = blacklist_tabix.query(region_chrom, query_start, query_end)
            for overlap in query_result:
                o_start = int(overlap[1])
                o_end = int(overlap[2])
                # mask the correspoinding DNA sequence as 'N'
                rev_start = o_start - sequence_start
                rev_end = o_end - sequence_start
                if rev_start > 0 and rev_start < len(region_seq) and rev_end > 0 and rev_end < len(region_seq):
                    region_seq = region_seq[:rev_start] + 'N' * (rev_end - rev_start) + region_seq[rev_end:]
        # create variant sequence 
        region_seq_var =  region_seq[:variant_rev_start] + row['allele2'] + region_seq[variant_rev_end:]
        # flip (reverse complement) the sequence if gene is on the downstream of region (ie, genomic coordinate is larger)
        if gene_start > region_end:
            region_seq = rcDNA(region_seq) 
            region_seq_var = rcDNA(region_seq_var)
        # padding N to the sequence if the sequence
        if len(region_seq) <= seq_len_cutoff:
            region_seq = region_seq + 'N' * (seq_len_cutoff - len(region_seq))
        else:
            region_seq = region_seq[:seq_len_cutoff]
        if len(region_seq_var) <= seq_len_cutoff:
            region_seq_var = region_seq_var + 'N' * (seq_len_cutoff - len(region_seq_var))
        else:
            region_seq_var = region_seq_var[:seq_len_cutoff]
        N_pass +=1
        # append to the dataset
        dataset.append((region_seq, region_seq_var, target))
    # report the statistics
    print("# Finish parsing eQTL records")
    print("# Total records: ", N_total)
    print("# Skipped records due to different chromosomes: ", N_skip_for_chrom_diff)
    print("# Skipped records due to distance cutoff: ", N_skip_for_len)
    print("# Skipped records due to unknown strand: ", N_skip_for_strand)
    print("# Select records %s with subset %s " % (N_pass, subset_group_name))
    return dataset





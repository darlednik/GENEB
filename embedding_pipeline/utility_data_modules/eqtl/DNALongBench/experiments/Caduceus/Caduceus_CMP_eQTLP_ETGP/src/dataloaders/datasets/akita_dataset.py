import tensorflow as tf
import torch
import numpy as np
import os
import json
from natsort import natsorted
import glob
import sys

tf.config.set_visible_devices([], "GPU")

train_data_path = "data_long_range_dna/Akita/tfrecords/train-*.tfr"

SEQUENCE_LENGTH = 1048576
TARGET_LENGTH = 99681


class AkitaDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfr_pattern, cell_type):
        super(AkitaDataset).__init__()
        self.dataset = self.read_tfr(tfr_pattern)
        self.cell_type = cell_type
        # original Akita data
        # target_ind_dict = {'HFF': 0, 'H1hESC': 1, 'GM12878': 2, 'IMR90': 3, 'HCT116': 4}
        # new extra data 
        target_ind_dict = {"HAP1": 0,  "Hela": 1, "HUVEC": 2, "K562": 3}
        self.target_ind = target_ind_dict[self.cell_type]


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
        # num = 100
        for seq_raw, targets_raw in self.dataset:
            seq = seq_raw.cpu().numpy().reshape(-1, 4).astype('int8')
            targets = targets_raw.cpu().numpy().reshape(TARGET_LENGTH, -1).astype('float16')
            # 200 bin
            seq = seq[-475136: -65536, :]
            # 150 bin
            # seq = seq[-372736: -65536, :]
            # 100 bin
            # seq = seq[-270336: -65536, :]
            seq = np.argmax(seq, axis=-1)
            targets = targets[:, self.target_ind]
            
            # 200 bin
            targets = targets[-19701:]
            # 150 bin
            # targets = targets[-11026: ]
            # 100 bin
            # targets = targets[-4851: ]
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


def get_dataloader(data_path, cell_type):
    dataset = AkitaDataset(data_path, cell_type)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)
    return loader

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# for i, batch in enumerate(train_loader):
#     batch_gpu = {k:v.to(device) for k,v in batch.items()}
#     seq = batch_gpu['sequence']
#     target = batch_gpu['target']
#     seq = seq[:, 536576: -65536, :]
#     target = target[:, -23653: ]
#     print(seq.shape)
#     print(target.shape)
#     print(seq[0][0])
#     print(max(target[0]))
#     print(min(target[0]))
#     break
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
import selene_sdk
from collections import namedtuple
import random
import tabix
from abc import ABCMeta
# from selene_sdk.samplers.dataloader import SamplerDataLoader
from selene_sdk.targets import Target, GenomicFeatures
from selene_sdk.samplers import Sampler
from selene_sdk.utils import get_indices_and_probabilities
import pyBigWig
import logging

tf.config.set_visible_devices([], "GPU")

task_name = 'transcription_initiation_signal_prediction'
root = '/ocean/projects/bio240015p/shared/DNALongBench/'
batch_size = 1

def get_dataloader(task_name = 'transcription_initiation_signal_prediction', subset = None):  
    genome, noblacklist_genome = get_genomes(root+"transcription_initiation_signal_prediction/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    tfeature = GenomicSignalFeatures([root+"transcription_initiation_signal_prediction/targets/agg.plus.bw.bedgraph.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.encodecage.plus.v2.bedgraph.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.plus.v2.bedgraph.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.plus.grocap.bedgraph.sorted.merged.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.plus.allprocap.bedgraph.sorted.merged.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.minus.allprocap.bedgraph.sorted.merged.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.minus.grocap.bedgraph.sorted.merged.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.minus.v2.bedgraph.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.encodecage.minus.v2.bedgraph.bw",
    root+"transcription_initiation_signal_prediction/targets/agg.minus.bw.bedgraph.bw"],
                                    ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
    ,'encoderampage_minus', 'encodecage_minus','cage_minus'],
                                    (100000,),
                                    [root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.plus.bed.gz",root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.minus.bed.gz"],
                                    [0,9], [1,8], [0.61357, 0.61357])
    
    sampler = RandomPositionsSampler(
                    reference_sequence = genome,
                    target= tfeature,
                    features = [''],
                    test_holdout=['chr8', 'chr9'],
                    validation_holdout= ['chr10'],
                    sequence_length= 100000,
                    center_bin_to_predict= 100000,
                    position_resolution=1,
                    random_shift=0,
                    random_strand=False,
                    one_hot = False
    )
    sampler.mode="train"
    sample_loader = SamplerDataLoader(sampler, num_workers=0, batch_size=batch_size, seed=3, one_hot=False)
    max_samples = 100000
    subset = Subset(sample_loader.dataset, list(range(max_samples)))
    train_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True
    )

    validseq = noblacklist_genome.get_encoding_from_coords("chr10", 0, 114364328)
    validcage = tfeature.get_feature_data("chr10", 0, 114364328)

    class ValidDataset(Dataset):
        def __init__(self, seq, cage, window_size=100000, step_size=50000,one_hot=True):
            """
            seq: (N, 4) numpy array, the one-hot-encoded genomic sequence
            cage: (10, N) numpy array, the target features
            window_size: int, size of the sliding window
            step_size: int, step size for the sliding window
            """
            self.seq = seq
            self.cage = cage
            self.window_size = window_size
            self.step_size = step_size
            self.num_windows = (seq.shape[0] - window_size) // step_size + 1
            self.one_hot = one_hot

        def __len__(self):
            return self.num_windows

        def __getitem__(self, idx):
            """
            Returns a tuple (input_sequence, target_features) for the idx-th window.
            """
            start = idx * self.step_size
            end = start + self.window_size
            input_seq = self.seq[start:end, :]  # Shape: (window_size, 4)
            target_cage = self.cage[:, start:end]  # Shape: (10, window_size)
            if self.one_hot:
                return input_seq, target_cage
            else:
                return noblacklist_genome.encoding_to_sequence(input_seq), target_cage
    valid_dataset = ValidDataset(validseq, validcage, window_size=100000, step_size=50000,one_hot=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_subset = Subset(valid_dataset, list(range(max_samples)))
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader 

class _SamplerDataset(Dataset):
    """
    This class provides a Dataset interface that wraps around a Sampler.
    `_SamplerDataset` is used internally by `SamplerDataLoader`.

    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data.
    """
    def __init__(self, sampler, one_hot):
        super(_SamplerDataset, self).__init__()
        self.sampler = sampler
        self.one_hot = one_hot

    def __getitem__(self, index):
        """
        Retrieve sample(s) from self.sampler. Only index length affects the
        number of samples. The index values are not used.

        Parameters
        ----------
        index : int or any object with __len__ method implemented
            The size of index is used to determine the number of the
            samples to return.

        Returns
        ----------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`I \\times L \\times N`, where :math:`I` is
            `index`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`I \\times T`,
            where :math:`T` is the number of targets predicted.
        """
        sequences, targets = self.sampler.sample(
            batch_size=1 if isinstance(index, int) else len(index))

        # print('_SamplerDataset:',self.one_hot,len(sequences[0]))
        if self.one_hot:
            if sequences.shape[0] == 1:
                sequences = sequences[0,:]
                targets = targets[0,:]
            return sequences, targets
        else:
            targets = targets.squeeze()
            return sequences, targets

    def __len__(self):
        """
        Implementing __len__ is required by the DataLoader. So as a workaround,
        this returns `sys.maxsize` which is a large integer which should
        generally prevent the DataLoader from reaching its size limit.

        Another workaround that is implemented is catching the StopIteration
        error while calling `next` and reinitialize the DataLoader.
        """
        return sys.maxsize


class SamplerDataLoader(DataLoader):
    """
    A DataLoader that provides parallel sampling for any `Sampler` object.
    `SamplerDataLoader` can be used with `MultiSampler` by specifying
    the `SamplerDataLoader` object as `train_sampler`, `validate_sampler`
    or `test_sampler` when initiating a `MultiSampler`.

    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data.
    num_workers : int, optional
        Default to 1. Number of workers to use for DataLoader.
    batch_size : int, optional
        Default to 1. The number of samples the iterator returns in one step.
    seed : int, optional
        Default to 436. The seed for random number generators.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data. Specified by the `sampler` param.
    num_workers : int
        Number of workers to use for DataLoader.
    batch_size : int
        The number of samples the iterator returns in one step.

    """
    def __init__(self,
                 sampler,
                 num_workers=1,
                 batch_size=1,
                 seed=436, one_hot=True):
        def worker_init_fn(worker_id):
            """
            This function is called to initialize each worker with different
            numpy seeds (torch seeds are set by DataLoader automatically).
            """
            np.random.seed(seed + worker_id)

        args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn
        }

        super(SamplerDataLoader, self).__init__(_SamplerDataset(sampler, one_hot), **args)
        self.seed = seed

logger = logging.getLogger(__name__)
SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])

class OnlineSampler(Sampler, metaclass=ABCMeta):
    """
    A sampler in which training/validation/test data is constructed
    from random sampling of the dataset for each batch passed to the
    model. This form of sampling may alleviate the problem of loading an
    extremely large dataset into memory when developing a new model.

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    target : selene_sdk.targets.Target or str
        A `selene_sdk.targets.Target` object to provide the targets that
        we would like to predict, or a str to provide path to tabix-indexed,
        compressed BED file (`*.bed.gz`) of genomic coordinates mapped to
        the genomic features we want to predict. Using str as target will
        be deprecated in the future. Please consider using a GenomicFeatures
        object instead.
    features : list(str)
        List of distinct features that we aim to predict.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['X', 'Y']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0], optional
        Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object. Use str target and feature_thresholds
        is deprecated and will be removed in the future. Please consider 
        passing GenomicFeatures object directly to target instead.
    mode : {'train', 'validate', 'test'}, optional
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `[]` the empty list. The list of modes for which we should
        save the sampled data to file (e.g. `["test", "validate"]`).
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    bin_radius : int
        From the center of the sequence, the radius in which to detect
        a feature annotation in order to include it as a sample's label.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    Raises
    ------
    ValueError
            If `mode` is not a valid mode.
    ValueError
        If the parities of `sequence_length` and `center_bin_to_predict`
        are not the same.
    ValueError
        If `sequence_length` is smaller than `center_bin_to_predict` is.
    ValueError
        If the types of `validation_holdout` and `test_holdout` are not
        the same.

    """
    STRAND_SIDES = ('+', '-')
    """
    Defines the strands that features can be sampled from.
    """

    def __init__(self,
                 reference_sequence,
                 target,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1001,
                 center_bin_to_predict=201,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):

        """
        Creates a new `OnlineSampler` object.
        """
        super(OnlineSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed + 1)

        if (sequence_length + center_bin_to_predict) % 2 != 0:
            raise ValueError(
                "Sequence length of {0} with a center bin length of {1} "
                "is invalid. These 2 inputs should both be odd or both be "
                "even.".format(
                    sequence_length, center_bin_to_predict))

        surrounding_sequence_length = sequence_length - center_bin_to_predict
        if surrounding_sequence_length < 0:
            raise ValueError(
                "Sequence length of {0} is less than the center bin "
                "length of {1}.".format(
                    sequence_length, center_bin_to_predict))

        # specifying a test holdout partition is optional
        if test_holdout:
            self.modes.append("test")
            if isinstance(validation_holdout, (list,)) and \
                    isinstance(test_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self.test_holdout = [str(c) for c in test_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float) and \
                    isinstance(test_holdout, float):
                self.validation_holdout = validation_holdout
                self.test_holdout = test_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout and test holdout must have the "
                    "same type (list or float) but validation was "
                    "type {0} and test was type {1}".format(
                        type(validation_holdout), type(test_holdout)))
        else:
            self.test_holdout = None
            if isinstance(validation_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
            else:
                self.validation_holdout = validation_holdout

        if mode not in self.modes:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.modes, mode))
        self.mode = mode

        self.surrounding_sequence_radius = int(
            surrounding_sequence_length / 2)
        self.sequence_length = sequence_length
        self.bin_radius = int(center_bin_to_predict / 2)
        self._start_radius = self.bin_radius
        if center_bin_to_predict % 2 == 0:
            self._end_radius = self.bin_radius
        else:
            self._end_radius = self.bin_radius + 1

        self.reference_sequence = reference_sequence

        self.n_features = len(self._features)

        if isinstance(target, str):
            self.target = GenomicFeatures(
                target, self._features,
                feature_thresholds=feature_thresholds)
        elif isinstance(target, Target) or isinstance(target, list):
            self.target = target
        elif target is None:
            self.target = None
        else:
            raise ValueError("target must be one of str, "
            "selene_sdk.targets.Target object, list, or None")
            
        self._save_filehandles = {}

    def get_feature_from_index(self, index):
        """
        Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        index : int
            The index of the feature to retrieve the name for.

        Returns
        -------
        str
            The name of the feature occurring at the specified index.
        """
        return self.target.index_feature_dict[index]

    def get_sequence_from_encoding(self, encoding):
        """
        Gets the string sequence from the one-hot encoding
        of the sequence.

        Parameters
        ----------
        encoding : numpy.ndarray
            An :math:`L \\times N` array (where :math:`L` is the length
            of the sequence and :math:`N` is the size of the sequence
            type's alphabet) containing the one-hot encoding of the
            sequence.

        Returns
        -------
        str
            The sequence of :math:`L` characters decoded from the input.
        """
        return self.reference_sequence.encoding_to_sequence(encoding)

    def save_dataset_to_file(self, mode, close_filehandle=False):
        """
        Save samples for each partition (i.e. train/validate/test) to
        disk.

        Parameters
        ----------
        mode : str
            Must be one of the modes specified in `save_datasets` during
            sampler initialization.
        close_filehandle : bool, optional
            Default is False. `close_filehandle=True` assumes that all
            data corresponding to the input `mode` has been saved to
            file and `save_dataset_to_file` will not be called with
            `mode` again.
        """
        if mode not in self._save_datasets:
            return
        samples = self._save_datasets[mode]
        if mode not in self._save_filehandles:
            self._save_filehandles[mode] = open(
                os.path.join(self._output_dir,
                             "{0}_data.bed".format(mode)),
                'w+')
        file_handle = self._save_filehandles[mode]
        while len(samples) > 0:
            cols = samples.pop(0)
            line = '\t'.join([str(c) for c in cols])
            file_handle.write("{0}\n".format(line))
        if close_filehandle:
            file_handle.close()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `n_samples` is None and the mode is `validate`, will
            set `n_samples` to 32000; if the mode is `test`, will set
            `n_samples` to 640000 if it is None. If the mode is `train`
            you must have specified a value for `n_samples`.
        mode : str, optional
            Default is None. The mode to run the sampler in when
            fetching the samples. See
            `selene_sdk.samplers.IntervalsSampler.modes` for more
            information. If None, will use the current mode `self.mode`.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        if mode is not None:
            self.set_mode(mode)
        else:
            mode = self.mode
        sequences_and_targets = []
        if n_samples is None and mode == "validate":
            n_samples = 32000
        elif n_samples is None and mode == "test":
            n_samples = 640000

        n_batches = int(n_samples / batch_size)
        for _ in range(n_batches):
            inputs, targets = self.sample(batch_size)
            sequences_and_targets.append((inputs, targets))
        targets_mat = np.vstack([t for (s, t) in sequences_and_targets])
        if mode in self._save_datasets:
            self.save_dataset_to_file(mode, close_filehandle=True)
        return sequences_and_targets, targets_mat

    def get_dataset_in_batches(self, mode, batch_size, n_samples=None):
        """
        This method returns a subset of the data for a specified run
        mode, divided into mini-batches.

        Parameters
        ----------
        mode : {'test', 'validate'}
            The mode to run the sampler in when fetching the samples.
            See `selene_sdk.samplers.IntervalsSampler.modes` for more
            information.
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of samples to retrieve.
            If `None`, it will retrieve 32000 samples if `mode` is validate
            or 640000 samples if `mode` is test or train.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            The list is length :math:`S`, where :math:`S =` `n_samples`.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`

        """
        return self.get_data_and_targets(
            batch_size, n_samples=n_samples, mode=mode)

    def get_validation_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 32000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        return self.get_dataset_in_batches(
            "validate", batch_size, n_samples=n_samples)

    def get_test_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 640000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.


        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        if "test" not in self.modes:
            raise ValueError("No test partition of the data was specified "
                             "during initialization. Cannot use method "
                             "`get_test_set`.")
        return self.get_dataset_in_batches("test", batch_size, n_samples)







class RandomPositionsSampler(OnlineSampler):
    """This sampler randomly selects a position in the genome and queries for
    a sequence centered at that position for input to the model.

    TODO: generalize to selene_sdk.sequences.Sequence?

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        A reference sequence from which to create examples.
    target : sselene_sdk.targets.Target or list(selene_sdk.targets.Target) or str
        A `selene_sdk.targets.Target` object to provide the targets that
        we would like to predict, or a list of these objects,
        or a str to provide path to tabix-indexed,
        compressed BED file (`*.bed.gz`) of genomic coordinates mapped to
        the genomic features we want to predict. Using str as target will
        be deprecated in the future. Please consider using a GenomicFeatures
        object instead.
    features : list(str)
        List of distinct features that we aim to predict.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['chrX', 'chrY']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0], optional
        Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object. Use str target and feature_thresholds
        is deprecated and will be removed in the future. Please consider 
        passing GenomicFeatures object directly to target instead.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `['test']`. The list of modes for which we should
        save the sampled data to file.
    position_resolution : int, optional
        Default is 1. Random coordinates will be rounded to multiples 
        of position_resolution. This can be useful for example
        when target stores binned data.
    random_strand : bool, optional
        Default is True. If True, sequences are retrieved randomly
        from positive or negative strand, otherwise the positive
        strand is used by default. Note that random_strand should be
        set to False if target provides strand-specific data.
    random_shift : int, optional
        Default is 0. If True, the coordinates to retrieve sequence
        are shifted by a random integer from -random_shift to 
        random_shift independently for each sample.  
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    bin_radius : int
        From the center of the sequence, the radius in which to detect
        a feature annotation in order to include it as a sample's label.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    position_resolution : int
        Default is 1. Random coordinates will be rounded to multiples 
        of position_resolution. This can be useful for example
        when target stores binned data.
    random_strand : bool
        Default is True. If True, sequences are retrieved randomly
        from positive or negative strand, otherwise the positive
        strand is used by default. Note that random_strand should be
        set to False if target provides strand-specific data.
    random_shift : int
        Default is 0. If True, the coordinates to retrieve sequence
        are shifted by a random integer from -random_shift to 
        random_shift independently for each sample.  
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    """
    def __init__(self,
                 reference_sequence,
                 target,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1000,
                 center_bin_to_predict=200,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 position_resolution=1,
                 random_shift=0,
                 random_strand=True,
                 output_dir=None,
                 one_hot=True):
        super(RandomPositionsSampler, self).__init__(
            reference_sequence,
            target,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []
        self.initialized = False
        self.position_resolution = position_resolution
        self.random_shift= random_shift
        self.random_strand=random_strand
        self.one_hot = one_hot

    def init(func):
        #delay initlization to allow  multiprocessing
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                if self._holdout_type == "chromosome":
                    self._partition_genome_by_chromosome()
                else:
                     self._partition_genome_by_proportion()

                for mode in self.modes:
                    self._update_randcache(mode=mode)
                self.initialized = True
            return func(self, *args, **kwargs)
        return dfunc


    def _partition_genome_by_proportion(self):
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom)
        n_intervals = len(self.sample_from_intervals)

        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_genome_by_chromosome(self):
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        for index, (chrom, len_chrom) in enumerate(self.reference_sequence.get_chr_lens()):
            if chrom in self.validation_holdout:
                self._sample_from_mode["validate"].indices.append(
                    index)
            elif self.test_holdout and chrom in self.test_holdout:
                self._sample_from_mode["test"].indices.append(
                    index)
            else:
                self._sample_from_mode["train"].indices.append(
                    index)

            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom - 2 * self.sequence_length)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = \
                self._sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

    def _retrieve(self, chrom, position, one_hot=True):
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        if self.target is not None:
            if isinstance(self.target, list):
                retrieved_targets = [t.get_feature_data(
                        chrom, bin_start, bin_end) for t in self.target]
            else:
                retrieved_targets = self.target.get_feature_data(
                    chrom, bin_start, bin_end)
            if retrieved_targets is None:
                logger.info("Target returns None. Sampling again.".format(
                                chrom, position))
                return None
        else:
            retrieved_targets = None


        window_start = bin_start - self.surrounding_sequence_radius
        window_end = bin_end + self.surrounding_sequence_radius
        if window_end - window_start < self.sequence_length:
            print(bin_start, bin_end,
                  self._start_radius, self._end_radius,
                  self.surrounding_sequence_radius)
            return None
        if self.random_strand:
            strand = self.STRAND_SIDES[random.randint(0, 1)]
        else:
            strand = '+'
            
        if self.random_shift > 0:
            r = np.random.randint(-self.random_shift, self.random_shift)
        else:
            r = 0
        retrieved_seq = \
            self.reference_sequence.get_encoding_from_coords(
                chrom, window_start+r, window_end+r, strand)
        if not self.one_hot:
            seq_string = self.reference_sequence.encoding_to_sequence(retrieved_seq) ############################################
            # print(seq_string[:10],len(seq_string))
        if retrieved_seq.shape[0] == 0 or retrieved_seq.shape[0] != self.sequence_length:
            logger.info("Full sequence centered at {0} position {1} "
                        "could not be retrieved. Sampling again.".format(
                            chrom, position))
            return None
        elif np.mean(retrieved_seq==0.25) >0.30: 
            logger.info("Over 30% of the bases in the sequence centered "
                        "at {0} position {1} are ambiguous ('N'). "
                        "Sampling again.".format(chrom, position))
            return None



        if self.mode in self._save_datasets and not isinstance(retrieved_targets, list):
            feature_indices = ';'.join(
                [str(f) for f in np.nonzero(retrieved_targets)[0]])
            self._save_datasets[self.mode].append(
                [chrom,
                 window_start,
                 window_end,
                 strand,
                 feature_indices])
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)
        
        if self.one_hot:
            # print('one hot', retrieved_seq.shape)
            return (retrieved_seq, retrieved_targets)
        else:
            # print('not one hot')
            return (seq_string, retrieved_targets)

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=200000,
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    @init
    def sample(self, batch_size=1, mode=None, return_coordinates=False, coordinates_only=False):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
            
        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        mode = mode if mode else self.mode
        if coordinates_only:
            assert return_coordinates == True
        else:
            if self.one_hot:
                sequences = np.zeros((batch_size, self.sequence_length, 4))
            else:
                sequences = [None] * batch_size             
            
            if self.target is None:
                targets = None
            elif isinstance(self.target, list):
                targets = [np.zeros((batch_size, *t.shape)) for t in self.target]
            elif isinstance(self.target.shape, list):
                targets = [np.zeros((batch_size, *tshape)) for tshape in self.target.shape]
            else:
                targets = np.zeros((batch_size, *self.target.shape))
        if return_coordinates:
            coords = []

        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[mode]["sample_next"]
            if sample_index == len(self._randcache[mode]["cache_indices"]):
                self._update_randcache()
                sample_index = 0

            rand_interval_index = \
                self._randcache[mode]["cache_indices"][sample_index]
            self._randcache[mode]["sample_next"] += 1

            chrom, cstart, cend = \
                self.sample_from_intervals[rand_interval_index]
            position = np.random.randint(cstart, cend)
            position -= position % self.position_resolution
            
            if not coordinates_only:
                retrieve_output = self._retrieve(chrom, position)
                if not retrieve_output:
                    continue

            if return_coordinates:
                coords.append((chrom, position))

            if not coordinates_only:
                seq, seq_targets = retrieve_output
                if self.one_hot:
                    sequences[n_samples_drawn, :, :] = seq
                else:
                    sequences[n_samples_drawn] = seq
                if isinstance(targets, list):
                    assert isinstance(seq_targets, (list, tuple))
                    for target, seq_target in zip(targets, seq_targets):
                        target[n_samples_drawn, :] = seq_target
                elif targets is not None:
                    targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
            

        if return_coordinates:
            if coordinates_only:
                return coords
            else:
                if target is None:
                    return sequences, coords
                else:
                    return sequences, targets, coords
        else:
            if targets is None:
                return sequences,
            else:
                return sequences, targets

def get_genomes(path):
    genome = selene_sdk.sequences.Genome(
                    input_path=path,
                    blacklist_regions= 'hg38'
                )
    noblacklist_genome = selene_sdk.sequences.Genome(
                    input_path=path )
    return genome, noblacklist_genome

class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

            
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat
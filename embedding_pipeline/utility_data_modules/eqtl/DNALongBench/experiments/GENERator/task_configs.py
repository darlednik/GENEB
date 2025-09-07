import torch
import torch.nn as nn
import torch.optim as optim
import dnalongbench
from dnalongbench.utils import load_data, BasenjiDataSet
import numpy as np
import tensorflow as tf
from model import RegulatorySignalPredictor, LongSequenceRegressionModel, LongSequenceClassificationModel, EqtlModel


def get_model(args):
    task_name = args.task_name
    if task_name == 'contact_map_prediction':
        model = LongSequenceRegressionModel(
            base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
            num_labels=40000,
            max_subsequence_length=8534,
            num_subsequences=8
        )
    elif task_name == 'regulatory_sequence_activity':
        if args.subset == 'human':
            model = RegulatorySignalPredictor(
                base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
                max_subsequence_length=4096,   # e.g. 32770/8 ≈ 4096
                num_subsequences=8,
                output_bins=896,
                output_tracks=5313
            )

        else: # 'mouse'
            model = RegulatorySignalPredictor(
                base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
                max_subsequence_length=4096,   
                num_subsequences=8,
                output_bins=896,
                output_tracks=1643
            )
    elif task_name == 'transcription_initiation_signal_prediction':
        model = RegulatorySignalPredictor(
            base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
            max_subsequence_length=2084,   
            num_subsequences=8,
            output_bins=10,
            output_tracks=100000
        )
    elif task_name == 'enhancer_target_gene_prediction':
        model = LongSequenceClassificationModel(
            base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
            num_labels=2,
            max_subsequence_length=9375,
            num_subsequences=8
        )
    elif task_name == 'eqtl_prediction':
        model = EqtlModel(
            base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
            num_labels=2,
            max_subsequence_length=9375,
            num_subsequences=8
        )
    return model










from evo2 import Evo2
import argparse 
import csv
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from scipy.stats import pearsonr
import selene_sdk
from scipy.signal import convolve
from selene_sdk.targets import Target
from datasets.tisp_dataset import get_dataloader, get_genomes
import pyBigWig
import tabix





task_name = 'transcription_initiation_signal_prediction'
root = '/ocean/projects/bio240015p/shared/DNALongBench/'
batch_size = 1

save_dir = "/ocean/projects/bio240015p/wcheng5/DNALongBench/experiments/Evo2_CMP_ETGP_eQTLP/Evo2_RSAP_TISP/results/TISP"

genome, noblacklist_genome = get_genomes(root+"transcription_initiation_signal_prediction/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")

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



def preprocess():
    print("Preprocessing TISP test data")
    # tfeature = GenomicSignalFeatures([root+"agg.plus.bw.bedgraph.bw",
    # root+"agg.encodecage.plus.v2.bedgraph.bw",
    # root+"agg.encoderampage.plus.v2.bedgraph.bw",
    # root+"agg.plus.grocap.bedgraph.sorted.merged.bw",
    # root+"agg.plus.allprocap.bedgraph.sorted.merged.bw",
    # root+"agg.minus.allprocap.bedgraph.sorted.merged.bw",
    # root+"agg.minus.grocap.bedgraph.sorted.merged.bw",
    # root+"agg.encoderampage.minus.v2.bedgraph.bw",
    # root+"agg.encodecage.minus.v2.bedgraph.bw",
    # root+"agg.minus.bw.bedgraph.bw"],
    #                             ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
    # ,'encoderampage_minus', 'encodecage_minus',
    # 'cage_minus'],
    #                             (100000,),
    #                             [root+"fantom.blacklist8.plus.bed.gz",root+"fantom.blacklist8.minus.bed.gz"],
    #                             [0,9], [1,8], [0.61357, 0.61357])
    
    # genome, noblacklist_genome = get_genomes(root+"transcription_initiation_signal_prediction/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
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
    



    allseq1 = genome.get_encoding_from_coords('chr8',0, 145138636)
    allcage1 = tfeature.get_feature_data('chr8',0, 145138636)
    allcage1 = allcage1[:,25000:-25000]
    allseq1 = allseq1[25000:-25000,:]
    print('allcage1',allcage1.shape)
    print('allseq1',allseq1.shape)

    save_path = os.path.join(save_dir, 'allcage.npy')
    np.save(save_path, allcage1)
    print(f"Done saving {save_path}")

    save_path = os.path.join(save_dir, 'allseq.npy')
    np.save(save_path, allseq1)
    print(f"Done saving {save_path}")


    allseq2 = genome.get_encoding_from_coords('chr9',0, 138394717)
    allcage2 = tfeature.get_feature_data('chr9',0, 138394717)
    allcage2 = allcage2[:,25000:-25000]
    allseq2 = allseq2[25000:-25000,:]
    print('allcage2',allcage2.shape)
    print('allseq2',allseq2.shape)

    save_path = os.path.join(save_dir, 'allcage2.npy')
    np.save(save_path, allcage2)
    print(f"Done saving {save_path}")

    save_path = os.path.join(save_dir, 'allseq2.npy')
    np.save(save_path, allseq2)
    print(f"Done saving {save_path}")


    allseq = np.concatenate([allseq1, allseq2], axis=0)

    allseq_n = (allseq == 0.25).all(axis=-1)
    allseq_n_1k = convolve(allseq_n, np.ones(1001), mode='same')
    valid_seqlocs = allseq_n_1k<0.1

    print('valid_seqlocs.shape',valid_seqlocs.shape)

    save_path = os.path.join(save_dir, 'valid_seqlocs.npy')
    np.save(save_path, valid_seqlocs)
    print(f"Done saving {save_path}")


    allcage = np.concatenate([allcage1, allcage2], axis=1)
    allcage[:,~valid_seqlocs]=0
    print('allcage.shape',allcage.shape)

    
    save_path = os.path.join(save_dir, 'allcage_valid.npy')
    np.save(save_path, allcage)
    print(f"Done saving {save_path}")


   
class SlidingWindowDataset(Dataset):
    def __init__(self, allseq, window_size, step, one_hot):
        """
        allseq: 2D array of shape (L, D)
        window_size: number of positions per window (e.g. 100_000)
        step: sliding step size (e.g. 50_000)
        """
        self.allseq      = allseq
        self.window_size = window_size
        self.step        = step
        L = allseq.shape[0]
        self.starts = list(range(0, L - window_size + 1, step))
        self.one_hot = one_hot
    
    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, idx):
        i   = self.starts[idx]
        seq = self.allseq[i : i + self.window_size, :]        # shape (window_size, D)
        # print('seq',seq.shape)
        fwd = torch.FloatTensor(seq[None, :, :])
        rev = torch.FloatTensor(seq[::-1, ::-1].copy()[None, :, :])
        # print('fwd',fwd.shape)
        
        fwd = fwd.detach().cpu().numpy().squeeze()
        # print('fwd',fwd.shape)
        rev = rev.detach().cpu().numpy().squeeze()
        if self.one_hot:
            return fwd, rev
        else:
            return genome.encoding_to_sequence(fwd), genome.encoding_to_sequence(rev)
      


def test_chr8():
    
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
    



    allseq1 = genome.get_encoding_from_coords('chr8',0, 145138636)
    allcage1 = tfeature.get_feature_data('chr8',0, 145138636)  
    ds1 = SlidingWindowDataset(allseq1, window_size=100_000, step=50_000, one_hot=False)

    test_loader = DataLoader(
        ds1,
        batch_size=1,      
        shuffle=False,     
        num_workers=0,     # or more, depending on your CPU cores
        pin_memory=True
    )
    for batch in test_loader:
        fwd, rev = batch
        print(len(fwd[0]),len(rev[0]))
        break
    

    model = Evo2('evo2_7b')

    task_layer = nn.Sequential(
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10 * 100000),
        ).to("cuda")
            
    checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location="cpu")
    task_layer.load_state_dict(checkpoint ["state_dict"])
    task_layer.eval()  # or model.train() depending on use


    N_windows = len(ds1)
    C = 10
    L = 100000

    print('N_windows of ds1: ', N_windows)

    # create a memmap on disk for 2*N_windows entries of shape (C, L)
    memmap_path = os.path.join(save_dir, "all_pre_raw1.dat")
    all_pre_raw = np.memmap(
        memmap_path, 
        dtype=np.float32, 
        mode="w+", 
        shape=(2 * N_windows, C, L)
    )

    # stream predictions into that memmap, interleaving fwd/rev
    
    idx_ptr = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            fwd, rev = batch


            fwd_input_ids = torch.tensor(
                model.tokenizer.tokenize(fwd[0]),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            layer_name = 'blocks.28.mlp.l3'

            with autocast(device_type="cuda", dtype=torch.bfloat16):  # or torch.float16 if unsupported
                _, embeddings = model(fwd_input_ids, return_embeddings=True, layer_names=[layer_name])
                hidden = torch.mean(embeddings[layer_name], dim=1) # [1, 4096]
                fwd_logits = task_layer(hidden.float())   # [1, 2]
                del embeddings, hidden
                torch.cuda.empty_cache()
            
            rev_input_ids = torch.tensor(
                model.tokenizer.tokenize(rev[0]),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            layer_name = 'blocks.28.mlp.l3'
            with autocast(device_type="cuda", dtype=torch.bfloat16):  # or torch.float16 if unsupported
                _, embeddings = model(rev_input_ids, return_embeddings=True, layer_names=[layer_name])
                hidden = torch.mean(embeddings[layer_name], dim=1) # [1, 4096]
                rev_logits = task_layer(hidden.float())   # [1, 2]
                del embeddings, hidden
                torch.cuda.empty_cache()

            # forward
            out_fwd = fwd_logits
            out_fwd = out_fwd.float().cpu().reshape(out_fwd.size(0), 10, 100000).numpy()  # (B, C, L)
 
            # reverse
            out_rev = rev_logits
            out_rev = out_rev.float().cpu().reshape(out_rev.size(0), 10, 100000).numpy()  # (B, C, L)

            B = out_fwd.shape[0]
            # build an interleaved block of shape (2*B, C, L)
            block = np.empty((2*B, C, L), dtype=np.float32)
            block[0::2] = out_fwd
            block[1::2] = out_rev

            # write to memmap
            start_idx = idx_ptr
            end_idx   = idx_ptr + 2*B
            all_pre_raw[start_idx:end_idx] = block

            print(f"Saved batch {batch_idx} → memmap indices [{start_idx}:{end_idx})")

            idx_ptr = end_idx

    # flush to disk
    all_pre_raw.flush()
    print("Predictions written to", memmap_path)


    allseq2 = genome.get_encoding_from_coords('chr9',0, 138394717)
    allcage2 = tfeature.get_feature_data('chr9',0, 138394717)

    ds2 = SlidingWindowDataset(allseq2, window_size=100_000, step=50_000, one_hot=False)

    test_loader2 = DataLoader(
        ds2,
        batch_size=1,      
        shuffle=False,     
        num_workers=0,     # or more, depending on your CPU cores
        pin_memory=True
    )
    for batch in test_loader2:
        fwd, rev = batch
        print(len(fwd[0]),len(rev[0]))
        break
    
    

def test_chr9():
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
    



    allseq2 = genome.get_encoding_from_coords('chr9',0, 138394717)
    allcage2 = tfeature.get_feature_data('chr9',0, 138394717)  
    ds2 = SlidingWindowDataset(allseq2, window_size=100_000, step=50_000, one_hot=False)

    test_loader2 = DataLoader(
        ds2,
        batch_size=1,      
        shuffle=False,     
        num_workers=0,     # or more, depending on your CPU cores
        pin_memory=True
    )
    for batch in test_loader2:
        fwd, rev = batch
        print(len(fwd[0]),len(rev[0]))
        break
    

    model = Evo2('evo2_7b')

    task_layer = nn.Sequential(
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10 * 100000),
        ).to("cuda")
            
    checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location="cpu")
    task_layer.load_state_dict(checkpoint ["state_dict"])
    task_layer.eval()  # or model.train() depending on use

    N_windows = len(ds2)
    C = 10
    L = 100000
    print('N_windows of ds2: ', N_windows)

    # create a memmap on disk for 2*N_windows entries of shape (C, L)
    memmap_path = os.path.join(save_dir, "all_pre_raw2_v2.dat")
    all_pre_raw2 = np.memmap(
        memmap_path, 
        dtype=np.float32, 
        mode="w+", 
        shape=(2 * N_windows, C, L)
    )

    # stream predictions into that memmap, interleaving fwd/rev
    
    idx_ptr = 0
    with torch.no_grad():
        # for batch_idx, batch in enumerate(test_loader2):
        for batch_idx, batch in tqdm(enumerate(test_loader2, 1),
                            desc="Eval",
                            unit="step",
                            ncols=80,
                            leave=True):
            fwd, rev = batch


            fwd_input_ids = torch.tensor(
                model.tokenizer.tokenize(fwd[0]),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            layer_name = 'blocks.28.mlp.l3'

            with autocast(device_type="cuda", dtype=torch.bfloat16):  # or torch.float16 if unsupported
                _, embeddings = model(fwd_input_ids, return_embeddings=True, layer_names=[layer_name])
                hidden = torch.mean(embeddings[layer_name], dim=1) # [1, 4096]
                fwd_logits = task_layer(hidden.float())   # [1, 2]
                del embeddings, hidden
                torch.cuda.empty_cache()
            
            rev_input_ids = torch.tensor(
                model.tokenizer.tokenize(rev[0]),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            layer_name = 'blocks.28.mlp.l3'
            with autocast(device_type="cuda", dtype=torch.bfloat16):  # or torch.float16 if unsupported
                _, embeddings = model(rev_input_ids, return_embeddings=True, layer_names=[layer_name])
                hidden = torch.mean(embeddings[layer_name], dim=1) # [1, 4096]
                rev_logits = task_layer(hidden.float())   # [1, 2]
                del embeddings, hidden
                torch.cuda.empty_cache()

            # forward
            out_fwd = fwd_logits
            out_fwd = out_fwd.float().cpu().reshape(out_fwd.size(0), 10, 100000).numpy()  # (B, C, L)
 
            # reverse
            out_rev = rev_logits
            out_rev = out_rev.float().cpu().reshape(out_rev.size(0), 10, 100000).numpy()  # (B, C, L)

            B = out_fwd.shape[0]
            # build an interleaved block of shape (2*B, C, L)
            block = np.empty((2*B, C, L), dtype=np.float32)
            block[0::2] = out_fwd
            block[1::2] = out_rev

            # write to memmap
            start_idx = idx_ptr
            end_idx   = idx_ptr + 2*B
            all_pre_raw2[start_idx:end_idx] = block
       
            print(f"Saved batch {batch_idx} → memmap indices [{start_idx}:{end_idx})")

            idx_ptr = end_idx

    # flush to disk
    all_pre_raw2.flush()
    print("Predictions written to", memmap_path)


def process():
    N_windows, C, L = (2901, 10, 100000)
    allpred_raw = np.memmap(
        os.path.join(save_dir, 'all_pre_raw1.dat'),
        dtype=np.float32,
        mode='r',
        shape=(2 * N_windows, C, L)
    )
    print("allpred_raw.shape", allpred_raw.shape)   # (11338, 100000, 10)
    if (allpred_raw[0]<0).any():
        print("Use exp to convert logit to probability")
        allpred_raw = np.exp(allpred_raw)   # (5802, 10, 100000)
    pointer = 0
    allpred1 = np.zeros((10,145138636))
    for i in np.arange(0, 145138636, 50000)[:-2]:
        pred1 = allpred_raw[pointer] #.transpose(1,0)   # (10, 100_000)
        pred2 = allpred_raw[pointer+1] #.transpose(1,0)
        if np.isnan(pred1).any():
            print(i, "pred1", pred1.min(), pred1.max())
            pred1 = np.nan_to_num(pred1)
        if np.isnan(pred2).any():
            print(i, "pred2", pred2.min(), pred2.max())
            pred2 = np.nan_to_num(pred2)
            # pred1[np.isnan(pred2)] = 0
        allpred1[:,i+25000:i+75000] = pred1[:,25000:75000]*0.5 + pred2[:,25000:75000]*0.5
        pointer += 2
   
    save_path = os.path.join(save_dir, 'allpre1.npy') 
    np.save(save_path, allpred1)
    print("Save allpred1")
    


    N_windows, C, L = (2766, 10, 100000)
    allpred_raw = np.memmap(
        os.path.join(save_dir, 'all_pre_raw2_v2.dat') ,
        dtype=np.float32,
        mode='r',
        shape=(2 * N_windows, C, L)
    )
    print("allpred_raw2.shape", allpred_raw.shape)
    if (allpred_raw[0]<0).any():
        print("Use exp to convert logit to probability")
        allpred_raw = np.exp(allpred_raw)
    allpred2 = np.zeros((10,138394717))
    pointer = 0
    for i in np.arange(0, 138394717, 50000)[:-2]:
        pred1 = allpred_raw[pointer] #.transpose(1,0)   # (10, 100_000)
        pred2 = allpred_raw[pointer+1] #.transpose(1,0)
        if np.isnan(pred1).any():
            print(i, "pred1", pred1.min(), pred1.max())
            pred1 = np.nan_to_num(pred1)
        if np.isnan(pred2).any():
            print(i, "pred2", pred2.min(), pred2.max())
            pred2 = np.nan_to_num(pred2)
        allpred2[:,i+25000:i+75000] = pred1[:,25000:75000]*0.5 + pred2[:,25000:75000]*0.5
        pointer += 2
    save_path = os.path.join(save_dir, 'allpred2.npy')
    np.save(save_path, allpred2)
    print("Save allpred2")

    allpred1 = np.load(os.path.join(save_dir, 'allpred1.npy'))
    allpred2 = np.load(os.path.join(save_dir, 'allpred2.npy'))
    allpred1 = allpred1[:,25000:-25000]
    allpred2 = allpred2[:,25000:-25000]
    print("allpred1.shape", allpred1.shape, "allpred2.shape",allpred2.shape)
    allpred = np.concatenate([allpred1, allpred2], axis=1)
    print('allpred.shape',allpred.shape)
  

    valid_seqlocs = np.load(os.path.join(save_dir, 'valid_seqlocs.npy') )
    print('valid_seqlocs.shape',valid_seqlocs.shape)

    allpred[:,~valid_seqlocs]=0
    save_path = os.path.join(save_dir, 'allpred.npy')
    np.save(save_path, allpred)
    print(f"Done saving {save_path}")

def eval():
    allcage = np.load(os.path.join(save_dir, 'allcage_valid.npy'), mmap_mode='r')
    allpred = np.load(os.path.join(save_dir, 'allpred.npy'), mmap_mode='r')
    valid_seqlocs = np.load(os.path.join(save_dir, 'valid_seqlocs.npy') )
    print(allcage.shape)
    print(allpred.shape)
    print(valid_seqlocs.shape)
    
    cor1 = pearsonr(np.log(np.concatenate([allpred[0,valid_seqlocs],allpred[-1,valid_seqlocs]])+1e-6), 
             np.log(np.concatenate([allcage[0,valid_seqlocs],allcage[-1,valid_seqlocs]])+1e-6))[0]
    print('FANTOM CAGE: ', cor1)
    cor2 = pearsonr(np.log(np.concatenate([allpred[1,valid_seqlocs],allpred[-2,valid_seqlocs]])+1e-6), 
                np.log(np.concatenate([allcage[1,valid_seqlocs],allcage[-2,valid_seqlocs]])+1e-6))[0]
    print('ENCODE CAGE: ', cor2)
    cor3 = pearsonr(np.log(np.concatenate([allpred[2,valid_seqlocs],allpred[-3,valid_seqlocs]])+1e-6), 
                np.log(np.concatenate([allcage[2,valid_seqlocs],allcage[-3,valid_seqlocs]])+1e-6))[0]
    print('ENCODE RAMPAGE: ', cor3)
    cor4 = pearsonr(np.log(np.concatenate([allpred[3,valid_seqlocs],allpred[-4,valid_seqlocs]])+1e-6), 
                np.log(np.concatenate([allcage[3,valid_seqlocs],allcage[-4,valid_seqlocs]])+1e-6))[0]
    print('GRO-cap: ', cor4)
    cor5 = pearsonr(np.log(np.concatenate([allpred[4,valid_seqlocs],allpred[-5,valid_seqlocs]])+1e-6), 
                np.log(np.concatenate([allcage[4,valid_seqlocs],allcage[-5,valid_seqlocs]])+1e-6))[0]
    print('PRO-cap: ', cor5)



if __name__ == "__main__":

    if not os.path.exists(os.path.join(save_dir, 'valid_seqlocs.npy')):
        print('Run preprocess')
        preprocess()
    if not os.path.exists(os.path.join(save_dir, 'all_pre_raw1.dat')):
        print('Test chr8')
        test_chr8()
    if not os.path.exists(os.path.join(save_dir, 'all_pre_raw2.dat')):
        print('Test chr9')
        test_chr9()
    if not os.path.exists(os.path.join(save_dir, 'allpred.npy')):
        print('Test chr9')
        process()
    eval()

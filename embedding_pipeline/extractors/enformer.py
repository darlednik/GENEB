from enformer_pytorch import Enformer
from typing import List
from tqdm import tqdm

import numpy as np
import torch

class EnformerPyTorchExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        # load pretrained Enformer v1 hyperparams
        self.model = Enformer.from_hparams(
            dim=1536,
            depth=11,
            heads=8,
            output_heads=dict(human=5313, mouse=1643),
            target_length=896
        ).to(self.device)
        self.model.eval()
        self.seq_length = 393216

    @staticmethod
    def one_hot_seq(seq: str, seq_length: int):
        """
        Static helper to convert a sequence to one-hot tensor.

        Args:
            seq: Input nucleotide sequence.
            seq_length: Fixed length for cropping/padding and encoding.

        Returns:
            One-hot encoded tensor of shape (4, seq_length).
        """
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        # uppercase, replace U->T
        seq = seq.upper().replace('U', 'T')
        # center crop or pad
        L = len(seq)
        if L >= seq_length:
            start = (L - seq_length)//2
            sub = seq[start:start+seq_length]
        else:
            pad = seq_length - L
            left = pad//2
            sub = 'N'*left + seq + 'N'*(pad-left)
        # one-hot
        arr = np.zeros((4, seq_length), dtype=np.float32)
        for i, c in enumerate(sub):
            if c in mapping:
                arr[mapping[c], i] = 1.0
        return torch.tensor(arr)

    def extract_embeddings(self, sequences: List[str], batch_size=1):
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Each sequence is one-hot encoded, passed through the Enformer model,
        and the 'human' output track is averaged over spatial and channel dimensions.

        Inputs:
            sequences: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once (mem-efficient).

        Returns:
            NumPy array of shape (len(sequences),) containing one embedding per sequence.
        """
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, 3, batch_size), desc="Extracting"):
                batch = sequences[i:i+batch_size]
                # prepare batch tensor (b,4,L)
                tensor = torch.stack([self.one_hot_seq(s, self.seq_length) for s in batch], dim=0).permute(0, 2, 1)
 
                tensor = tensor.to(self.device)
                out = self.model(tensor)
                human = out['human']  # shape (b,896,5313)
                # mean pool spatial dims and channels -> (b,)
                print(human.shape)
                emb = human.mean(dim=(2)).cpu().numpy()
                embs.append(emb)
        
        print(np.concatenate(embs, axis=1).shape)
        print(np.concatenate(embs, axis=1))
        return np.concatenate(embs, axis=1)

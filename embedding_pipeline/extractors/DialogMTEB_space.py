import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import List

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(DIR, "utility_modules/SPACE"))

from model.config_space import SpaceConfig
from model.modeling_space import Space, TrainingSpace

TrainingSpace.config_class = SpaceConfig


class SPACEExtractor:
    def __init__(self, name_model: str, device: str='cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.name_model = name_model
        self.model = Space.from_pretrained(name_model).to(self.device).eval()

        self.seq_length = 131_072

    @staticmethod
    def one_hot_seq(seq: str, seq_length: int = 131072):
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        seq = seq.upper().replace('U', 'T')

        L = len(seq)
        if L >= seq_length:
            start = (L - seq_length) // 2
            sub = seq[start:start+seq_length]
        else:
            pad = seq_length - L
            left = pad // 2
            sub = 'N'*left + seq + 'N'*(pad - left)

        arr = np.zeros((seq_length, 4), dtype=np.float32)

        for i, c in enumerate(sub):
            if c in mapping:
                arr[i, mapping[c]] = 1.0

        return torch.from_numpy(arr)  # (L, 4), dtype=float32
    
    def extract_embeddings(self, sequences: List[str], batch_size=1) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Each sequence is one-hot encoded, passed through the SPACE model,
        and the 'human' output track is averaged over spatial and channel dimensions.

        Args:
            sequences: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once (mem-efficient).
        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """

        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i:i+batch_size]

                arrs = [self.one_hot_seq(s, self.seq_length) for s in batch]

                tensor = torch.stack(arrs, dim=0).to(self.device)

                output = self.model(tensor, return_embeddings=True)
                
                emb = output[-1].mean(dim=1).detach().cpu().numpy()

                all_embs.append(emb)
        return np.vstack(all_embs)

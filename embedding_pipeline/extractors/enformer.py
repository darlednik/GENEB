from enformer_pytorch import from_pretrained
from typing import List
from tqdm import tqdm

import numpy as np
import torch

class EnformerPyTorchExtractor:
    def __init__(self, name_model: str, device: str='cpu'):
        self.device = device or torch.device(device if torch.cuda.is_available() else 'cpu')
  
        self.model = from_pretrained(
            name_model, 
            use_tf_gamma=False
        )
        self.model.to(self.device).eval()
        
        config = self.model.config

        self.seq_length = 196_608
        self.target_length = config.target_length

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
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        seq = seq.upper().replace('U', 'T')

        L = len(seq)
        if L >= seq_length:
            start = (L - seq_length) // 2
            sub = seq[start:start + seq_length]
        else:
            pad = seq_length - L
            left = pad // 2
            sub = 'N' * left + seq + 'N' * (pad - left)

        arr = np.zeros((seq_length, 4), dtype=np.float32)
        for i, c in enumerate(sub):
            if c in mapping:
                arr[i, mapping[c]] = 1.0
        return torch.tensor(arr)

    def extract_embeddings(self, sequences: List[str], batch_size=1) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Each sequence is one-hot encoded, passed through the Enformer model,
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
                batch = sequences[i:i + batch_size]

                arrs = [self.one_hot_seq(s, self.seq_length) for s in batch]

                tensor = torch.stack(arrs, dim=0).to(self.device)

                human = self.model(tensor)['human']

                emb = human.mean(dim=1).cpu().numpy()

                all_embs.append(emb)
                
        return np.concatenate(all_embs, axis=0)

from typing import List, Optional, Literal
from tqdm import tqdm
from .base import BaseEmbeddingExtractor
import numpy as np
import torch
from enformer_pytorch import from_pretrained


class EnformerPyTorchExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        name_model: str,
        device: str = "cpu",
    ):
        
        self.device = device
        self.seq_length = int(seq_length)

        self.model = from_pretrained(
            name_model,
            use_tf_gamma=False,
        )
        self.model.to(self.device)
        self.model.eval()

        self._map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    @staticmethod
    def reverse_complement(seq: str) -> str:
        comp = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A",
                              "a": "t", "c": "g", "g": "c", "t": "a",
                              "N": "N", "n": "n"})
        return seq.translate(comp)[::-1]

    def seq_to_indices(self, seq: str, length: int) -> torch.Tensor:
        """
        Convert DNA sequence string to indices in {0...4} for A,C,G,T,N.
        Pads with 'N' or truncates to length.
        """
        s = seq.upper()
        if len(s) >= length:
            s = s[:length]
        else:
            s = s + ("N" * (length - len(s)))

        arr = torch.empty(length, dtype=torch.long)
        for i, ch in enumerate(s):
            arr[i] = self._map.get(ch, 4)  # unknown -> N
        return arr

    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Args:
            sequences: list of DNA strings
            batch_size: batching
        Returns:
            np.ndarray shape (len(sequences), 3072)
        """
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i:i + batch_size]

                x = torch.stack([self.seq_to_indices(s, self.seq_length) for s in batch], dim=0).to(self.device)

                # Forward pass with embeddings
                _, emb = self.model(x, return_embeddings=True)  # emb: (B, 896, 3072)
                pooled = emb.mean(dim=1)  # (B, 3072)

                all_embs.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(all_embs, axis=0)


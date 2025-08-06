from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np
import torch


class GroverExtractor:
    def __init__(self, name_model: str, device: str='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.model = AutoModelForMaskedLM.from_pretrained(
            name_model, output_hidden_states=True
        ).to(device)
        self.model.eval()

    def extract_embeddings(self, sequences: list[str], batch_size: int = 1) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Args:
            sequences: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once (mem-efficient).

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting embs...'):
            batch_seqs = sequences[i : i + batch_size]
            enc = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**enc)
                hidden = out.hidden_states[-1]

            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            pooled = summed / counts

            all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs)
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import numpy as np
import torch


class DNABERTSExtractor:
    def __init__(self, name_model: str, device: str='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, use_fast=True)
        self.model = AutoModel.from_pretrained(name_model).to(device)
        self.model.eval()

        self.max_length = getattr(self.model.config, 'max_position_embeddings', 512)

    def extract_embeddings(self, sequences: List[str], batch_size=1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
            batch = sequences[i: i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
                cls_emb = out[:, 0, :]
            all_embs.append(cls_emb.cpu().numpy())

        return np.vstack(all_embs)
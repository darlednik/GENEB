from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import numpy as np
import torch

class DNABERT2Extractor:
    def __init__(self, name_model: str, device: str='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name_model, trust_remote_code=True, return_dict=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
            batch = sequences[i : i + batch_size]
            enc = self.tokenizer(batch,
                                  padding=True,
                                  truncation=True,
                                  return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            cls_embeds = out[0][:, 0, :].cpu().numpy()
            all_embeds.append(cls_embeds)
        return np.vstack(all_embeds)
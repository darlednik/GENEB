from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .base import BaseEmbeddingExtractor
import numpy as np
import torch

class DNABERT2Extractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name_model, trust_remote_code=True, return_dict=True)
        self.device = device
        self.max_length = 512
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
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]
                enc = self.tokenizer(batch,
                                      padding=True,
                                      truncation=True,
                                      max_length=self.max_length,
                                      return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)

                
                last_hidden = out[0]
                mask = enc['attention_mask'].unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                emb = (summed / counts)
                all_embeds.append(emb.cpu().numpy())
        return np.vstack(all_embeds)

from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseEmbeddingExtractor
import numpy as np
import torch


class MetageneExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.name_model = name_model

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_model, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_model,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
            return_dict=True
        )
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size=1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """

        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting embs...'):
                batch_seqs = sequences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                out = self.model(**inputs)
                last_hidden = out.hidden_states[-1]
                embs = last_hidden.mean(dim=1)

                embs = embs.to(torch.float32)

                all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)

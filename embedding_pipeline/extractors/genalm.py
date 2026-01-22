from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .base import BaseEmbeddingExtractor
import numpy as np
import torch

class GENALMExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.device = device 
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.model = AutoModel.from_pretrained(name_model, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = 512 # check HF

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
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
                batch_seqs = sequences[i:i + batch_size]
                encoded = self.tokenizer(batch_seqs,
                                         return_tensors='pt',
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                
                outputs = self.model.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state

                
                mask = attention_mask.unsqueeze(-1)
                sum_emb = (last_hidden * mask).sum(1)
                lengths = mask.sum(1)
                emb = (sum_emb / lengths).cpu().numpy()

                all_embs.append(emb)

        return np.vstack(all_embs)

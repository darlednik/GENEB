from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import numpy as np
import torch

class GENALMExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.model = AutoModel.from_pretrained(name_model, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch
            pooling: 'mean' for average pooling, 'cls' for CLS token

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting embs...'):
            batch_seqs = sequences[i:i + batch_size]
            encoded = self.tokenizer(batch_seqs,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state

                
                mask = attention_mask.unsqueeze(-1)
                sum_emb = (last_hidden * mask).sum(1)
                lengths = mask.sum(1)
                emb = (sum_emb / lengths).cpu().numpy()

            all_embs.append(emb)

        return np.vstack(all_embs)
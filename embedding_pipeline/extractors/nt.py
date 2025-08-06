from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np
import torch

class NucleotideTransformerExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(name_model, trust_remote_code=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = self.tokenizer.model_max_length

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
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting embs...'):
                batch = sequences[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden = outputs.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1)
                sum_hidden = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                seq_emb = (sum_hidden / lengths).cpu().numpy()
                all_embs.append(seq_emb)
        return np.vstack(all_embs)
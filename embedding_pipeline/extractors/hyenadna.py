from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import BaseEmbeddingExtractor
import numpy as np
import torch

class HyenadnaExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.name_model = name_model
        self.device = device
        

        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_model,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            output_hidden_states=True
        )
        self.max_length = self.tokenizer.model_max_length
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch_seqs = sequences[i:i + batch_size]
                enc = self.tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {"input_ids": enc["input_ids"].to(self.model.device)}
                outputs = self.model(**inputs)
                
                hidden_states = outputs.hidden_states[-1]              # (B, L, H)
                mask = enc["attention_mask"].to(hidden_states.device).unsqueeze(-1)  # (B, L, 1)

                hs = hidden_states.to(torch.float32)
                embs = (hs * mask).sum(dim=1) / mask.sum(dim=1)
                embs = embs.cpu().numpy()
                all_embeddings.append(embs)

        return np.vstack(all_embeddings)

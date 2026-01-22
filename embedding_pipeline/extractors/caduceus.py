from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .base import BaseEmbeddingExtractor
import numpy as np
import torch

class CaduceusExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        
        
        self.model = AutoModelForMaskedLM.from_pretrained(name_model, output_hidden_states=True, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        

    def extract_embeddings(self, seqs: list[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size]
                    
                enc = self.tokenizer(
                    batch,
                    add_special_tokens=False,     
                    padding=True,
                    truncation=True,              
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                last_hidden = outputs.hidden_states[-1]  # (B, L, H)
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)

                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                embeds = (summed / counts).float().cpu().numpy()

                all_embeds.append(embeds) 
        return np.vstack(all_embeds)

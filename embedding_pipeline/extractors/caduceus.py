from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np
import torch

class CaduceusExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        
        
        self.model = AutoModelForMaskedLM.from_pretrained(name_model, output_hidden_states=True, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        

    def extract_embeddings(self, seqs: list[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch
            pooling: 'mean' for average pooling, 'cls' for CLS token

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size]
                enc = self.tokenizer(batch,
                                        padding=True,
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")

                inputs = {"input_ids": enc["input_ids"].to(self.device)}
               
                
                outputs = self.model(**inputs)
                last_hidden = outputs.hidden_states[-1]
                mask = enc["attention_mask"].to(self.device).unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                embeds = (summed / counts).cpu().numpy()
                all_embeds.append(embeds)
        return np.vstack(all_embeds)

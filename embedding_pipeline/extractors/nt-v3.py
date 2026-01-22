from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from .base import BaseEmbeddingExtractor


class NucleotideTransformerExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name_model, trust_remote_code=True)
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        self.max_length = self.tokenizer.model_max_length
        self.pad_to_multiple_of = 128

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]

                enc = self.tokenizer(
                    batch,
                    add_special_tokens=False,     
                    return_tensors="pt",
                    padding=True,                 
                    truncation=True,
                    max_length=self.max_length,           
                    pad_to_multiple_of=self.pad_to_multiple_of,       
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                hidden = outputs.hidden_states[-1]  # (B, L, H)
                mask = attention_mask.unsqueeze(-1) # (B, L, 1)
                sum_hidden = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)

                seq_emb = (sum_hidden / lengths).float().cpu().numpy()
                all_embs.append(seq_emb)

        return np.vstack(all_embs)

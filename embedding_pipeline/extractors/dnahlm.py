from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
import torch

class DNAHLMExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, 
                                                       trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                name_model,
                output_hidden_states=True,
                return_dict=True,
                trust_remote_code=True
            )
            .to(self.device)
        )
        self.model.eval()

        

    def extract_embeddings(self, seqs: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Args:
            seqs (List[str]): List of nucleotide sequences.
            batch_size (int, optional): Number of sequences to process at once. Defaults to 1.

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                   # add_special_tokens=True
                )
                input_ids = enc.input_ids.to(self.device)
                attention_mask = enc.attention_mask.to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1)
                summed = (hidden_states * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = (summed / lengths).cpu().numpy()
                all_embeddings.append(emb)
        return np.vstack(all_embeddings)

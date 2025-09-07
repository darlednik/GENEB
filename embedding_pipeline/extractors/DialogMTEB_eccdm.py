import torch
from transformers import AutoTokenizer
from utility_modules.GenAI_Lab_project.BiMambaForMaskedLM import BiMambaForMaskedLM
from typing import List
import numpy as np
from tqdm import tqdm


class ECCDNAMambaExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = BiMambaForMaskedLM.from_pretrained(name_model, trust_remote_code=True)
        if hasattr(self.model, "mamba_forward"):
            self.model.mamba_forward.use_mem_eff_path = False
        self.model.to(self.device).eval()

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
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
            batch = sequences[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                hs = out.hidden_states
                mask = attention_mask.unsqueeze(-1)
                summed = (hs * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                pooled = summed / lengths
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs)
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import numpy as np
import torch

class OmniNAExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            name_model,
            output_hidden_states=True
        ).to(self.device)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.eval()

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
                batch = sequences[i: i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                out = self.model(**enc)

                hidden = out.hidden_states[-1]

                mask = enc.attention_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                embs = (summed / lengths).cpu().numpy()
                all_embs.append(embs)
        return np.vstack(all_embs)
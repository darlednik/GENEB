from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from .base import BaseEmbeddingExtractor

class MutBERTExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        name_model,
        device = "cpu",
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(name_model)

        self.model = AutoModel.from_pretrained(name_model, trust_remote_code = True).to(self.device)
        self.model.eval()

        self.vocab_size = len(self.tokenizer)

    def extract_embeddings(
        self,
        seqs: List[str],
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Returns:
            embeddings: np.ndarray [N, H]
        """
        outs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size]

                enc = self.tokenizer(
                    batch,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                input_ids = enc["input_ids"].to(self.device)             # [B, L]
                attention_mask = enc["attention_mask"].to(self.device)   # [B, L]

                mut_inputs = F.one_hot(input_ids, num_classes=self.vocab_size).to(torch.float32)

                mut_inputs = mut_inputs * attention_mask.unsqueeze(-1).to(mut_inputs.dtype)

                out = self.model(mut_inputs, attention_mask=attention_mask, return_dict=True)

                last_hidden = out.last_hidden_state  # [B, L, H]

                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, L, 1]
                summed = (last_hidden * mask).sum(dim=1)                   # [B, H]
                denom = mask.sum(dim=1).clamp(min=1)                       # [B, 1]
                embs = summed / denom
                outs.append(embs.to(torch.float32).cpu().numpy())

        return np.vstack(outs)

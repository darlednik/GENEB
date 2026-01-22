import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from .base import BaseEmbeddingExtractor

class GeneratorExtractor(BaseEmbeddingExtractor):

    def __init__(self, name_model: str, device = "cpu"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_model,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            name_model,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.tokenizer.padding_side = "right"

        self.max_length = self.model.config.max_position_embeddings

        self.bos_token = self.tokenizer.bos_token
        

    @staticmethod
    def _trim_to_multiple_of_6(seq: str) -> str:
        n = (len(seq) // 6) * 6
        return seq[:n]

    def _preprocess_batch(self, batch: List[str]) -> List[str]:
        processed = []
        for s in batch:
            s2 = self._trim_to_multiple_of_6(s)
            processed.append(self.bos_token + s2 if s2 else self.bos_token)
        return processed

    def extract_embeddings(self, seqs: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            seqs: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once.

        Returns:
            np.ndarray of shape (len(seqs), hidden_size)
        """

        all_embs = []


        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size]
                processed = self._preprocess_batch(batch)

                tok_kwargs = dict(
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding=True,
		            max_length=self.max_length,
                    truncation=True,
                )

                enc = self.tokenizer(processed, **tok_kwargs)
                enc = {k: v.to(self.device) for k, v in enc.items()}

                out = self.model(**enc, return_dict=True, output_hidden_states=True)

                last_hidden = out.hidden_states[-1]  # [B, L, H]

                mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # [B, L, 1]

                summed = (last_hidden * mask).sum(dim=1)  # [B, H]
                counts = mask.sum(dim=1).clamp(min=1)     # [B, 1]
                emb = summed / counts                     # [B, H]

                all_embs.append(emb.detach().to(torch.float32).cpu().numpy())

        return np.vstack(all_embs)

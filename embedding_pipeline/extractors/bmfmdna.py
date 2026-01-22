from typing import List, Optional, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import bmfm_target
from .base import BaseEmbeddingExtractor
from bmfm_targets.models.model_utils import register_configs_and_models


_REGISTERED = False

class BMFMDNAExtractor(BaseEmbeddingExtractor):

    def __init__(self, name_model, device = "cpu"):
        self.device = device
        
        global _REGISTERED

        if not _REGISTERED:
            register_configs_and_models()
            _REGISTERED = True

        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name_model, trust_remote_code=True, return_dict=True)

        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Args:
            sequences: list of nucleotide sequences (strings)
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]

                tok_kwargs: Dict[str, Any] = dict(
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                enc = self.tokenizer(batch, **tok_kwargs)
                enc = {k: v.to(self.device) for k, v in enc.items()}

                out = self.model(**enc)
                

                last_hidden = out.last_hidden_state

                attn = enc.get("attention_mask")

                mask = attn.unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                emb = summed / counts

                all_embeds.append(emb.to(torch.float32).cpu().numpy())

        return np.vstack(all_embeds)


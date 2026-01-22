from typing import List, Optional, Tuple
import os
from .base import BaseEmbeddingExtractor

import numpy as np
import torch
from tqdm import tqdm

import tokenizers
from utility_modules.deepgene.DeepGene.PanGeneGraphTrans.modeling_roformer import RoFormerForMaskedLM
from pathlib import Path

class DeepGeneExtractor(BaseEmbeddingExtractor):

    def __init__(
        self,
        name_model: str,
	    device = 'cpu'
    ):
        self.device = device
	
        tokenizer_path = Path(name_model) / "tokenizer.json"
        path_to_model  = Path(name_model)

        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

        self.model = RoFormerForMaskedLM.from_pretrained(path_to_model, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        self.cls_id = 1
        self.sep_id = 2
        self.pad_id = 3

        self.max_position_embeddings = self.model.config.max_position_embeddings

    def _encode_batch(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids_list = []
        max_len_raw = 0
        for s in seqs:
            ids = self.tokenizer.encode(s, add_special_tokens=False).ids
            if len(ids) + 2 >= self.max_position_embeddings:
                ids = ids[: self.max_position_embeddings - 2]
            ids_list.append(ids)
            max_len_raw = max(max_len_raw, len(ids))

        L = min(max_len_raw + 2, self.max_position_embeddings)  # total length with CLS+SEP

        bsz = len(ids_list)
        input_ids = torch.full((bsz, L), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, L), dtype=torch.bool)
        pos_ids = torch.full((bsz, L), L - 1, dtype=torch.long)  # PAD positions set to L-1

        for i, ids in enumerate(ids_list):
            n = min(len(ids), L - 2)

            # DeepGene code uses [CLS] at 0, [SEP] at 1, sequence starts at 2
            input_ids[i, 0] = self.cls_id
            input_ids[i, 1] = self.sep_id
            if n > 0:
                input_ids[i, 2 : 2 + n] = torch.tensor(ids[:n], dtype=torch.long)

            attention_mask[i, : 2 + n] = True

            pos_ids[i, 0] = 0
            pos_ids[i, 1] = 1

            if n > 0:
                pos_ids[i, 2 : 2 + n] = torch.arange(2, 2 + n, dtype=torch.long)

        return input_ids, attention_mask, pos_ids

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []
        
        with torch.no_grad():
            for start in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[start : start + batch_size]
                input_ids, attention_mask, pos_ids = self._encode_batch(batch)

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                pos_ids = pos_ids.to(self.device)

                out = self.model.roformer(
                    input_ids=input_ids,
                    pos_ids=pos_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                last_hidden = out.last_hidden_state  # [B, L, H]

                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, L, 1]
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                emb = (summed / counts).to(torch.float32).cpu().numpy()

                all_embeds.append(emb)

        return np.vstack(all_embeds)


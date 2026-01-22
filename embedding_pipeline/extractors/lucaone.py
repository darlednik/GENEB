from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .base import BaseEmbeddingExtractor

class LucaOneExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        name_model,
        device = "cpu"
    ):
        self.device = device
        self.seq_type = "gene" # "gene" for DNA/RNA, "prot" for protein
        self.add_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_model,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            name_model,
            task_level="token_level",
            task_type="embedding",
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()
        
        
        self.max_length = self.model.config.max_position_embeddings
        self.tokenize_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )

    @staticmethod
    def pad(x, val, max_len):
        return x + [val] * (max_len - len(x))

    def _encode_batch(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encs = []
        for seq in batch:
            enc = self.tokenizer.encode_plus(
                text=seq,
                seq_type=self.seq_type,
                add_special_tokens=self.add_special_tokens,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            encs.append(enc)

        max_len = max(len(e["input_ids"]) for e in encs)
        input_ids = torch.tensor(
            [self.pad(e["input_ids"], self.tokenizer.pad_token_id, max_len) for e in encs],
            dtype=torch.long
        )
        attention_mask = torch.tensor(
            [self.pad(e["attention_mask"], 0, max_len) for e in encs],
            dtype=torch.long,
        )
        return input_ids, attention_mask

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]
                input_ids, attention_mask = self._encode_batch(batch)

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
               

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()

                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                emb = emb.cpu().numpy()

                all_embs.append(emb)
        return np.vstack(all_embs)

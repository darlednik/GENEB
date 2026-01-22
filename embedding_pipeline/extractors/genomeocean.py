
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List 
from .base import BaseEmbeddingExtractor

DICT_CONFIG_MODELS = {
        "DOEJGI/GenomeOcean-4B": 10240,
        "DOEJGI/GenomeOcean-500M": 1024

}

class GenomeOceanExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        name_model: str,
        device = str,
    ):
        self.name_model = name_model
        self.device = device

        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_model,
            trust_remote_code=True,
            padding_side="left", 
        )

        self.max_seq_length = DICT_CONFIG_MODELS.get(self.name_model, 1024)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name_model,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )


        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of strings (DNA).
            batch_size: batch size for inference.

        Returns:
            np.ndarray of shape (N, hidden_size)
        """

        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting embs..."):
                batch = sequences[i : i + batch_size]

                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

                hidden = outputs.hidden_states[-1]  # (B, L, H)

                mask = attention_mask.unsqueeze(-1).type_as(hidden)  # (B, L, 1)
                summed = (hidden * mask).sum(dim=1)                  # (B, H)
                counts = mask.sum(dim=1).clamp(min=1.0)              # (B, 1)
                emb = summed / counts                                # (B, H)

                all_embs.append(emb.float().cpu().numpy())

        return np.vstack(all_embs)

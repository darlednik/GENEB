import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from pathlib import Path
import json
from tqdm import tqdm
from .base import BaseEmbeddingExtractor
from utility_modules.JanusDNA.caduceus.tokenization_caduceus import CaduceusTokenizer
from utility_modules.JanusDNA.janusdna.configuration_janusdna import JanusDNAConfig
from utility_modules.JanusDNA.janusdna.modeling_janusdna import JanusDNAModel


class JanusDNAExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.device = device
        model_dir = Path(name_model)

        self.path_to_ckpt = next(model_dir.glob("*.ckpt"))
        self.path_to_config = next(model_dir.glob("*.json"))

        self.max_length = 1024

        self.tokenizer = CaduceusTokenizer(model_max_length=self.max_length, padding_side="right")

        raw = json.loads(self.path_to_config.read_text())
        cfg = raw["config"]

        cfg["bidirectional"] = str(cfg["bidirectional"]).strip().lower().strip(",") == "true"



        self.config = JanusDNAConfig(**cfg)

        self.model = JanusDNAModel(self.config).to(self.device)
        self.model.eval()

        if self.device.startswith("cuda"):
            self.model = self.model.to(dtype=torch.bfloat16)
        
        
        ckpt = torch.load(self.path_to_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]

        cleaned = {}
        for k, v in state.items():
            if k.startswith(("train_torchmetrics.", "val_torchmetrics.", "test_torchmetrics.")):
                continue
            if k.startswith("model.model."):
                k = k[len("model.model."):]
            elif k.startswith("model."):
                k = k[len("model."):]
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        self.model.final_mlp = nn.Identity()

    def extract_embeddings(self, seqs: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size]

                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=True,
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                attn = attention_mask.unsqueeze(-1).float()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                hs = outputs.last_hidden_state  # [B, L, H]

                pooled = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1.0)

                all_embs.append(pooled.cpu().float().numpy())

        return np.vstack(all_embs)


import torch
import numpy as np
from typing import List
from pathlib import Path
import json
from tqdm import tqdm

from utility_modules.JanusDNA.caduceus.tokenization_caduceus import CaduceusTokenizer
from utility_modules.JanusDNA.janusdna.configuration_janusdna import JanusDNAConfig
from utility_modules.JanusDNA.janusdna.modeling_janusdna import JanusDNAModel


class JanusDNAExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        for file in Path(name_model).iterdir():
            if file.suffix == ".ckpt":
                self.path_to_CKPT = file
            elif file.suffix == '.json':
                self.path_to_config = file 

        self.tokenizer = CaduceusTokenizer(model_max_length=262_144, padding_side="right")

        with open(self.path_to_config) as f:
            config = json.load(f)

        self.config = JanusDNAConfig(**{k: v for k, v in config.items() if k != "vocab_size"})
        self.config.vocab_size = int(config["config"].get("vocab_size", 8))
        self.config.pad_vocab_size_multiple = 1

        self.model = JanusDNAModel(self.config).to(device)

        self.model.eval()

    def extract_embeddings(self, seqs: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Args:
            seqs: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once.

        Returns:
            np.ndarray of shape (len(seqs), hidden_size)
        """
        all_embs = []

        with torch.no_grad():

            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i:i+batch_size]
                enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=True
            )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device).unsqueeze(-1).float()
            

                outputs = self.model(input_ids=input_ids, attention_mask=enc["attention_mask"].to(self.device), return_dict=True)
            
                hs = outputs.last_hidden_state  # [B, L, H]
            
                pooled = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1.0)
                       
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs)


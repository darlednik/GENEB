import re
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from utility_modules.genomicsfm_project.genomics_fm.configuration_bert import BertConfig
from utility_modules.genomicsfm_project.genomics_fm.bert_layers import BertForMaskedLM
import utility_modules.genomicsfm_project.genomics_fm.bert_layers as bert_layers_mod
from .base import BaseEmbeddingExtractor

bert_layers_mod.flash_attn_qkvpacked_func = None


class GenomicsFMExtractor(BaseEmbeddingExtractor):

    def __init__(self, name_model, device = "cpu"):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)

        config = BertConfig.from_pretrained(name_model)
        self.model = BertForMaskedLM.from_pretrained(name_model, config=config)

        self.model.to(self.device)
        self.model.eval()

        self.max_len = 512

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of DNA strings
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]

                batch = [re.sub(r"[^ACGT]", "N", s.strip().upper()) for s in batch]

                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                sequence_output, _ = self.model.bert(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc["token_type_ids"],
                )

                emb = sequence_output[:, 0]
                all_embeds.append(emb.detach().cpu().numpy())

        return np.vstack(all_embeds)

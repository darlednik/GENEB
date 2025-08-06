from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import numpy as np
import torch


class OmniDNAExtractor:
    
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(name_model, trust_remote_code=True, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name_model, config=config, trust_remote_code=True)
        self.model.to(self.device).eval()

    def extract_embeddings(self, sequences: list[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)


                outputs = self.model(**enc)

                hidden_states = outputs.hidden_states[-1]

                mask = enc.attention_mask.unsqueeze(-1)
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                embeds = (summed / counts).cpu().numpy()

                all_embeds.append(embeds)

        return np.vstack(all_embeds)
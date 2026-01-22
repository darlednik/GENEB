from typing import List
from tqdm import tqdm
import numpy as np
import torch
from .base import BaseEmbeddingExtractor
from utility_modules.EVO_project.evo.evo.models import Evo

class EVOExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device = 'cpu'):
        self.device = device
        evo_model = Evo(name_model)
        self.model, self.tokenizer = evo_model.model, evo_model.tokenizer

        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, seqs: list[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch
        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeds = []

        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i : i + batch_size][0]

                inputs_ids = torch.tensor(self.tokenizer.tokenize(batch), dtype=torch.int).to(self.device).unsqueeze(0)
                
                outputs = self.model(inputs_ids)
                
                embedds = outputs[-1].mean(dim=1).float().cpu().numpy()
                
                all_embeds.append(embedds)
        return np.vstack(all_embeds)

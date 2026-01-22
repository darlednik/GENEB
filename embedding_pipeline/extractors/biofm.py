
from typing import List, Optional
from .base import BaseEmbeddingExtractor
import numpy as np
import torch
from tqdm import tqdm
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder


class BioFMExtractor(BaseEmbeddingExtractor):

    def __init__(
        self,
        name_model: str,
        device = "cpu",
    ):
        self.name_model = name_model
        self.device = device
        self.torch_dtype = torch.bfloat16

        self.model = AnnotatedModel.from_pretrained(
            self.name_model,
            torch_dtype=self.torch_dtype,
        )
        self.model = self.model.to(self.device)

        self.tokenizer = AnnotationTokenizer.from_pretrained(self.name_model)
        self.embedder = Embedder(self.model, self.tokenizer)

        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of DNA strings (plain sequences)
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), embedding_dim)
        """

        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]

                embs_t = self.embedder.get_sequence_embeddings(batch)

                embs_np = embs_t.to(torch.float32).detach().cpu().numpy()
                
                all_embs.append(embs_np)

        return np.vstack(all_embs).astype(np.float32, copy=False)



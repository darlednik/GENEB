from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import torch

class HyenadnaExtractor:
    def __init__(self, name_model: str, device: str = 'cpu'):
        self.name_model = name_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 160_000

        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_model,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True
        )
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

                hidden_states = outputs.hidden_states[-1]

                embs = hidden_states.mean(dim=1).to(torch.float32)
                embs = embs.cpu().numpy()
            all_embeddings.append(embs)

        return np.vstack(all_embeddings)
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

class GeneratorExtractor:
    def __init__(self, name_model: str, device: str='cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(name_model, remote_trust_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(name_model, output_hidden_states=True, remote_trust_code=True).to(self.device)
        self.model.eval()

    def extract_embeddings(self, seqs: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Args:
            sequences: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once (mem-efficient).

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i:i+batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                out = self.model(**enc, return_dict=True)

                last_hidden = out.hidden_states[-1]

                
                mask = enc.attention_mask.unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                emb = summed / counts

                all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)
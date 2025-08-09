from typing import List
from tqdm import tqdm

import numpy as np
import torch
from pathlib import Path

from utility_modules.DNAGPT.dna_gpt.model import DNAGPT
from utility_modules.DNAGPT.dna_gpt.tokenizer import KmerTokenizer

class DNAGPTExtractor:
    @staticmethod
    def get_model_and_tokenizer(model_name: str):
        """
        Initialization DNAGPT model and KmerTokenizer.
        """
        special_tokens = (
            [str(i) for i in range(10)] + ['+', '-', '*', '/', '=', '&', '|', '!'] +
            ['M', 'B', 'P', 'R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V', 'W', 'Y', 'X', 'Z']
        )
        dynamic = False if model_name == 'dna_gpt0.1b_h' else True
        tokenizer = KmerTokenizer(6, special_tokens, dynamic)
        model = DNAGPT.from_name(model_name, vocab_size=len(tokenizer))
        return model, tokenizer

    @staticmethod
    def load_weights(model, tokenizer, weight_path: str, device, dtype):
        state = torch.load(weight_path, map_location='cpu')
        sd = state.get('model', state)
        model.load_state_dict(sd, strict=False)
        model.to(device=device, dtype=dtype).eval()
        return model, tokenizer

    def __init__(self, name_model: str, device: str='cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = Path(name_model).stem
        self.weights = name_model

        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32

        model, tokenizer = self.get_model_and_tokenizer(self.model_name)

        self.model, self.tokenizer = self.load_weights(model, tokenizer, self.weights, self.device, self.dtype)
        self.max_len = getattr(self.model, 'max_len', None)

    def extract_embeddings(self, seqs: List[str], batch_size: int = 1) -> np.ndarray:
        """
        
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """
        all_emb = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting embs..."):
                batch = seqs[i:i+batch_size]
                encoded = [
                    self.tokenizer.encode(seq, max_len=self.max_len, device=self.device)
                    for seq in batch
                ]
                mlen = max(len(ids) for ids in encoded)
                input_ids = torch.full((len(encoded), mlen), self.tokenizer.pad_id, device=self.device, dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                for j, ids in enumerate(encoded):
                    input_ids[j, :len(ids)] = torch.tensor(ids, device=self.device)
                    attention_mask[j, :len(ids)] = 1

                outputs = self.model(
                    input_ids,
                    attention_mask
                )

                hidden = outputs[-1]
                mask = attention_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = (summed / lengths).cpu().numpy()
                all_emb.append(emb)
        return np.vstack(all_emb)
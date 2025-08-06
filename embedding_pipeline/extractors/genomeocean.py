from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm
import numpy as np

class GenomeOceanExtractor:
    def __init__(
        self, name_model: str, device: str='cpu'):

        self.name_model = name_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_model,
            trust_remote_code=True,
            padding_side="left",
        )

        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        load_kwargs = {
            "pretrained_model_name_or_path": self.name_model,
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs).to(self.device)
        self.model.eval()

    def extract_embeddings(self, sequences: List[str], batch_size = 1) -> np.ndarray:
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
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
                batch = sequences[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                ).to(self.device)

                outputs = self.model.base_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1]

                
                mask = enc["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                emb = summed / counts

                all_embs.append(emb.cpu().numpy())

        return np.vstack(all_embs)
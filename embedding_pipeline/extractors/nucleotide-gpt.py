from pathlib import Path
from typing import List, Optional, Tuple
import importlib.util

import numpy as np

from .base import BaseEmbeddingExtractor
import jax
import jax.numpy as jnp
from .base import 


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NucleotideGPTConfig:
    d_model: int = 2048
    ffw_multiplier: int = 4
    query_heads: int = 8
    key_heads: int = 8
    num_layers: int = 12
    key_dim: int = 128
    vocab_size: int = 6
    max_seq_len: int = 8192
    causal: bool = True
    use_attn_kernel: bool = False


class NucleotideGPTExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        name_model: str,
        device = "cpu"
    ):
        self.checkpoint_dir = Path(name_model)
        self.cfg_ngpt = NucleotideGPTConfig()

        model_py = Path("./utility_modules/nucleotide_gpt") / "projects" / "bio" / "modelling" / "model.py"

        self.ngpt = _load_module_from_path("nucleotide_gpt_modelling_model", model_py)

        mesh = self.ngpt.create_mesh()
        rules = self.ngpt.mdl_parallel_rules

        self.device = device

        self.cfg = self.ngpt.Config(
            d_model=self.cfg_ngpt.d_model,
            ffw_multiplier=self.cfg_ngpt.ffw_multiplier,
            query_heads=self.cfg_ngpt.query_heads,
            key_heads=self.cfg_ngpt.key_heads,
            num_layers=self.cfg_ngpt.num_layers,
            key_dim=self.cfg_ngpt.key_dim,
            vocab_size=self.cfg_ngpt.vocab_size,
            max_seq_len=self.cfg_ngpt.max_seq_len,
            causal=self.cfg_ngpt.causal,
            use_attn_kernel=self.cfg_ngpt.use_attn_kernel,
            weight_dtype_at_rest=jnp.float32,
            active_weight_dtype=jnp.bfloat16,
            rules=rules,
            mesh=mesh,
            max_lr=3e-4,
            min_lr=1e-5,
            warmup_steps=50,
            total_steps=100000,
        )

        ckpt_path = Path(self.checkpoint_dir).expanduser().resolve()

        mngr = self.ngpt.make_mngr(path=str(ckpt_path))
        self.weights, _ = self.ngpt.load(mngr, self.cfg)

        self._forward = jax.jit(self.ngpt.forward)

        self._stoi = {"P": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        self._stoi.update({"a": 1, "c": 2, "g": 3, "t": 4, "n": 5})
        self.pad_id = 0

    def _tokenize_batch(
        self, seqs: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          x: int32 [B, L]
          segment_ids: int32 [B, L] with 1 for real tokens, 0 for padding
        """
        max_cap = int(self.cfg_ngpt.max_seq_len)
        toks = []
        lens = []
        for s in seqs:
            s = s[:max_cap]
            arr = [self._stoi.get(ch, 0) for ch in s]  # unknown -> padding
            toks.append(arr)
            lens.append(len(arr))

        L = max(lens) if lens else 0
        if L == 0:
            x = np.zeros((len(seqs), 1), dtype=np.int32)
            seg = np.zeros((len(seqs), 1), dtype=np.int32)
            return x, seg

        x = np.full((len(seqs), L), self.pad_id, dtype=np.int32)
        seg = np.zeros((len(seqs), L), dtype=np.int32)

        for i, arr in enumerate(toks):
            if not arr:
                continue
            x[i, : len(arr)] = np.asarray(arr, dtype=np.int32)
            seg[i, : len(arr)] = 1

        return x, seg

    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Args:
            sequences: list of str sequences
            batch_size: number of sequences per batch

        Returns:
            np.ndarray of shape (len(sequences), hidden_size)
        """

        all_embeds = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embs..."):
            batch = sequences[i : i + batch_size]
            x_np, seg_np = self._tokenize_batch(batch)

            x = jnp.asarray(x_np, dtype=self.jnp.int32)
            segment_ids = jnp.asarray(seg_np, dtype=self.jnp.int32)

            _, _, hidden = self._forward(x, segment_ids, self.weights, self.cfg_ngpt)

            mask = segment_ids.astype(jnp.float32)[..., None]  # [B,L,1]
            summed = (hidden * mask).sum(axis=1) # [B,D]
            denom = mask.sum(axis=1)
            denom = jnp.maximum(denom, 1.0)
            emb = summed / denom # [B,D]

            all_embeds.append(np.asarray(emb, dtype=np.float32))

        return np.concatenate(all_embeds, axis=0)


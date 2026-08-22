"""Frozen SpliceBERT.1024nt embeddings for the GENEB harness."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForMaskedLM

from .base import BaseEmbeddingExtractor


MAX_NUCLEOTIDES = 1024
WINDOW_STRIDE = 512
HIDDEN_SIZE = 512
_TOKEN_IDS = {"N": 5, "A": 6, "C": 7, "G": 8, "T": 9}
_IUPAC_AMBIGUOUS = frozenset("RYSWKMBDHV")
_ALLOWED_BASES = frozenset(_TOKEN_IDS) | _IUPAC_AMBIGUOUS | {"U"}


def _normalize_sequence(sequence: str) -> str:
    """Normalize one nucleotide sequence for SpliceBERT.

    Parameters
    ----------
    sequence
        Raw DNA or RNA sequence.

    Returns
    -------
    str
        Uppercase DNA sequence with ambiguity codes converted to ``N``.

    Raises
    ------
    TypeError
        If ``sequence`` is not a string.
    ValueError
        If the sequence contains a non-IUPAC character.
    """
    if not isinstance(sequence, str):
        raise TypeError("SpliceBERT sequences must be strings")
    normalized = sequence.upper()
    invalid = sorted(set(normalized) - _ALLOWED_BASES)
    if invalid:
        raise ValueError(f"Unsupported characters in sequence: {invalid}")
    return "".join(
        "T" if base == "U" else "N" if base in _IUPAC_AMBIGUOUS else base
        for base in normalized
    )


def _window_starts(
    length: int,
    window_size: int = MAX_NUCLEOTIDES,
    stride: int = WINDOW_STRIDE,
) -> list[int]:
    """Return sliding-window starts with a final end-anchored window.

    Parameters
    ----------
    length
        Raw sequence length.
    window_size
        Maximum number of nucleotides per window.
    stride
        Distance between regular window starts.

    Returns
    -------
    list[int]
        Zero-based starts covering the complete sequence.
    """
    if length < 0:
        raise ValueError("Sequence length cannot be negative")
    if window_size < 1 or stride < 1:
        raise ValueError("Window size and stride must be positive")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    end_start = length - window_size
    if starts[-1] != end_start:
        starts.append(end_start)
    return starts


def _encode_batch(sequences: Sequence[str]) -> tuple[Tensor, Tensor]:
    """Encode normalized nucleotide windows with dynamic PAD tokens.

    Parameters
    ----------
    sequences
        Normalized windows no longer than 1,024 nt.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Integer input IDs and attention mask.
    """
    if not sequences:
        raise ValueError("Cannot encode an empty batch")
    if any(len(sequence) > MAX_NUCLEOTIDES for sequence in sequences):
        raise ValueError(f"SpliceBERT windows cannot exceed {MAX_NUCLEOTIDES} nt")

    max_tokens = max(len(sequence) for sequence in sequences) + 2
    input_ids = torch.zeros((len(sequences), max_tokens), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        encoded = [_TOKEN_IDS[base] for base in sequence]
        token_count = len(encoded) + 2
        input_ids[row, 0] = 2
        if encoded:
            input_ids[row, 1 : token_count - 1] = torch.tensor(encoded)
        input_ids[row, token_count - 1] = 3
        attention_mask[row, :token_count] = 1
    return input_ids, attention_mask


def _masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool final hidden states over all non-PAD tokens.

    The attention mask includes both ``[CLS]`` and ``[SEP]``.

    Parameters
    ----------
    hidden
        Final-layer hidden states with shape ``(batch, tokens, hidden)``.
    attention_mask
        Mask with one for non-PAD tokens.

    Returns
    -------
    torch.Tensor
        Pooled embeddings with shape ``(batch, hidden)``.
    """
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def _inference_context(device: torch.device) -> ContextManager[object]:
    """Return BF16 autocast when supported and FP32 otherwise."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class SpliceBERTExtractor(BaseEmbeddingExtractor):
    """Extract mean-pooled embeddings from a local SpliceBERT.1024nt model."""

    def __init__(self, name_model: str, device: str) -> None:
        """Load the frozen SpliceBERT encoder.

        Parameters
        ----------
        name_model
            Local Hugging Face checkpoint directory.
        device
            PyTorch device, for example ``cpu`` or ``cuda``.
        """
        self.name_model = str(
            Path(os.path.expandvars(os.path.expanduser(name_model))).resolve()
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        load_kwargs: dict[str, object] = {"local_files_only": True}
        if self.device.type == "cuda":
            load_kwargs["attn_implementation"] = "sdpa"
        container = AutoModelForMaskedLM.from_pretrained(
            self.name_model,
            **load_kwargs,
        )
        self.model = container.bert.to(self.device).eval()
        self.model.config.output_hidden_states = False
        if int(self.model.config.max_position_embeddings) != MAX_NUCLEOTIDES + 2:
            raise ValueError(
                "Expected SpliceBERT.1024nt max_position_embeddings=1026, "
                f"found {self.model.config.max_position_embeddings}"
            )
        if int(self.model.config.hidden_size) != HIDDEN_SIZE:
            raise ValueError(
                f"Expected SpliceBERT hidden_size={HIDDEN_SIZE}, "
                f"found {self.model.config.hidden_size}"
            )

    def extract_embeddings(
        self,
        sequences: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Embed sequences, averaging equally across long-sequence windows.

        Parameters
        ----------
        sequences
            DNA or RNA sequences in source order.
        batch_size
            Maximum number of windows per model forward pass.

        Returns
        -------
        numpy.ndarray
            Float32 embeddings with shape ``(n_sequences, 512)``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not sequences:
            return np.empty((0, HIDDEN_SIZE), dtype=np.float32)

        normalized = [_normalize_sequence(sequence) for sequence in sequences]
        windows: list[str] = []
        owners: list[int] = []
        for sequence_index, sequence in enumerate(normalized):
            for start in _window_starts(len(sequence)):
                windows.append(sequence[start : start + MAX_NUCLEOTIDES])
                owners.append(sequence_index)

        sums = np.zeros((len(sequences), HIDDEN_SIZE), dtype=np.float64)
        counts = np.zeros(len(sequences), dtype=np.int64)
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                batch_windows = windows[start : start + batch_size]
                batch_owners = owners[start : start + batch_size]
                input_ids, attention_mask = _encode_batch(batch_windows)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with _inference_context(self.device):
                    hidden = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                    ).last_hidden_state
                    pooled = _masked_mean_pool(hidden, attention_mask)
                batch_embeddings = pooled.float().cpu().numpy()
                for owner, embedding in zip(batch_owners, batch_embeddings):
                    sums[owner] += embedding
                    counts[owner] += 1

        return (sums / counts[:, None]).astype(np.float32)

"""Frozen Carbon hybrid-tokenizer DNA embeddings for the GENEB harness."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbeddingExtractor


KMER_SIZE = 6
TAG_TOKENS = 2  # <dna> ... </dna>
_IUPAC_AMBIGUOUS = frozenset("RYSWKMBDHVN")
_ALLOWED_BASES = frozenset("ACGT") | _IUPAC_AMBIGUOUS | {"U"}
_PROBE_SEQUENCE = "ACGT" * 30  # 120 nt -> 20 six-mers + 2 tags
# Hugging Face commits the submitted scores were computed with. Model and tokenizer
# are both loaded with trust_remote_code, so both are pinned.
REVISIONS = {
    "HuggingFaceBio/Carbon-500M": "2ac078d4dd6c79b5c1255b24b5c64a1982d97c74",
    "HuggingFaceBio/Carbon-3B": "e1b2369c087f39769a2107704b887a1c8ae1f3b8",
    "HuggingFaceBio/Carbon-8B": "2361b060209e1ccba08691e1dc31a3b8eb9e6816",
}


def _normalize_sequence(sequence: str) -> str:
    """Normalize one nucleotide sequence for the Carbon tokenizer.

    Parameters
    ----------
    sequence
        Raw DNA or RNA sequence.

    Returns
    -------
    str
        Uppercase DNA sequence with ambiguity codes collapsed to ``N``.

    Raises
    ------
    TypeError
        If ``sequence`` is not a string.
    ValueError
        If the sequence contains a non-IUPAC character.
    """
    if not isinstance(sequence, str):
        raise TypeError("Carbon sequences must be strings")
    normalized = sequence.upper()
    invalid = sorted(set(normalized) - _ALLOWED_BASES)
    if invalid:
        raise ValueError(f"Unsupported characters in sequence: {invalid}")
    return "".join(
        "T" if base == "U" else "N" if base in _IUPAC_AMBIGUOUS else base
        for base in normalized
    )


def _window_starts(length: int, window_size: int, stride: int) -> list[int]:
    """Return sliding-window starts with a final end-anchored window.

    Starts are multiples of :data:`KMER_SIZE` so that every window is cut on a
    six-mer boundary. The hybrid tokenizer reads non-overlapping six-mers from
    offset zero, so an unaligned start would shift the reading frame and embed
    a different token sequence for the same nucleotides.

    Parameters
    ----------
    length
        Normalized sequence length in nucleotides.
    window_size
        Maximum number of nucleotides per window.
    stride
        Distance between regular window starts.

    Returns
    -------
    list[int]
        Zero-based, frame-aligned starts covering the complete sequence.
    """
    if length < 0:
        raise ValueError("Sequence length cannot be negative")
    if window_size < KMER_SIZE or stride < KMER_SIZE:
        raise ValueError(f"Window size and stride must be at least {KMER_SIZE}")
    if window_size % KMER_SIZE or stride % KMER_SIZE:
        raise ValueError(f"Window size and stride must be multiples of {KMER_SIZE}")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    # Align the end-anchored window down to a six-mer boundary; the overlap it
    # gains with the previous window is harmless, a frame shift is not.
    end_start = ((length - window_size) // KMER_SIZE) * KMER_SIZE
    if starts[-1] != end_start:
        starts.append(end_start)
    return starts


def _masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool final hidden states over all non-PAD tokens.

    The mask covers the ``<dna>`` and ``</dna>`` tag tokens as well as the
    six-mers, matching the sequence the model was trained to read.

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


class CarbonExtractor(BaseEmbeddingExtractor):
    """Extract mean-pooled embeddings from a frozen Carbon decoder."""

    def __init__(self, name_model: str, device: str) -> None:
        """Load the Carbon backbone and its hybrid DNA tokenizer.

        The language-model head is not loaded: ``AutoModel`` resolves the
        checkpoint to its bare decoder, which is all that frozen-embedding
        evaluation needs.

        Parameters
        ----------
        name_model
            Hugging Face repository id or local checkpoint directory. Known
            repository ids are loaded at the commit listed in ``REVISIONS``.
        device
            PyTorch device, for example ``cpu`` or ``cuda``.

        Raises
        ------
        RuntimeError
            If CUDA is requested but unavailable, or if the tokenizer does not
            six-mer encode a probe sequence as expected.
        """
        self.name_model = name_model
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        revision = REVISIONS.get(name_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_model,
            revision=revision,
            trust_remote_code=True,
        )
        if int(getattr(self.tokenizer, "k", 0)) != KMER_SIZE:
            raise RuntimeError(
                f"Expected a {KMER_SIZE}-mer hybrid tokenizer, found k="
                f"{getattr(self.tokenizer, 'k', None)}"
            )

        load_kwargs: dict[str, object] = {
            "revision": revision,
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            load_kwargs["attn_implementation"] = "sdpa"
        self.model = AutoModel.from_pretrained(name_model, **load_kwargs)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        self.model.config.output_hidden_states = False

        self.hidden_size = int(self.model.config.hidden_size)
        # Leave room for the two DNA tag tokens the tokenizer adds.
        self.max_nucleotides = (
            int(self.model.config.max_position_embeddings) - TAG_TOKENS
        ) * KMER_SIZE
        self.window_stride = max(
            KMER_SIZE, (self.max_nucleotides // 2 // KMER_SIZE) * KMER_SIZE
        )
        self._check_tokenizer()

    def _check_tokenizer(self) -> None:
        """Fail loudly if six-mer encoding is not what the model expects.

        The hybrid tokenizer only six-mer encodes text inside ``<dna>`` tags
        and otherwise falls back to the base BPE vocabulary. That fallback is
        silent, produces plausible embeddings and would invalidate the whole
        run, so confirm the token count on a known sequence before embedding
        anything.
        """
        encoded = self._encode_batch([_PROBE_SEQUENCE])
        expected = len(_PROBE_SEQUENCE) // KMER_SIZE + TAG_TOKENS
        found = int(encoded["input_ids"].shape[1])
        if found != expected:
            raise RuntimeError(
                f"Tokenizer produced {found} tokens for a {len(_PROBE_SEQUENCE)} nt "
                f"probe, expected {expected}. DNA six-mer encoding is not active."
            )

    def _encode_batch(self, windows: Sequence[str]) -> dict[str, Tensor]:
        """Six-mer encode and right-pad one batch of nucleotide windows."""
        if not windows:
            raise ValueError("Cannot encode an empty batch")
        return self.tokenizer(
            list(windows),
            padding=True,
            return_tensors="pt",
            auto_dna_tags=True,
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
            Float32 embeddings with shape ``(n_sequences, hidden_size)``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not sequences:
            return np.empty((0, self.hidden_size), dtype=np.float32)

        normalized = [_normalize_sequence(sequence) for sequence in sequences]
        windows: list[str] = []
        owners: list[int] = []
        for sequence_index, sequence in enumerate(normalized):
            starts = _window_starts(
                len(sequence), self.max_nucleotides, self.window_stride
            )
            for start in starts:
                windows.append(sequence[start : start + self.max_nucleotides])
                owners.append(sequence_index)

        # Group windows of similar length together so that right-padding inside
        # a batch stays cheap on task sets with mixed sequence lengths.
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]))

        sums = np.zeros((len(sequences), self.hidden_size), dtype=np.float64)
        counts = np.zeros(len(sequences), dtype=np.int64)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                encoded = self._encode_batch([windows[i] for i in chunk])
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                with _inference_context(self.device):
                    hidden = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                    ).last_hidden_state
                    pooled = _masked_mean_pool(hidden, attention_mask)
                batch_embeddings = pooled.float().cpu().numpy()
                for index, embedding in zip(chunk, batch_embeddings):
                    sums[owners[index]] += embedding
                    counts[owners[index]] += 1

        return (sums / counts[:, None]).astype(np.float32)

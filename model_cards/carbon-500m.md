# carbon-500m

- **Display name:** Carbon 500M
- **Parameters:** 511,894,528 (checkpoint total; input and output embeddings are tied)
- **Architecture:** decoder-only autoregressive transformer, Llama-style
  (`LlamaForCausalLM`), 28 layers, hidden size 1,024, RoPE, native context 8,192 tokens
- **Weights / URL:** https://huggingface.co/HuggingFaceBio/Carbon-500M
- **Tokenizer / input:** `HybridDNATokenizer` — non-overlapping 6-mers for DNA plus the
  Qwen3 BPE vocabulary for text, 155,776 entries. DNA must be wrapped in `<dna>...</dna>`
  or it falls back silently to BPE; the extractor passes `auto_dna_tags=True` and asserts
  the resulting token count on a probe sequence before embedding anything. One token
  carries 6 bp, so the 8,192-token context covers ~49 kbp and no GENEB task (longest
  4,000 nt) needs windowing. Any 6-mer containing a non-`ACGT` base becomes `<oov>`.
- **Pooling:** mean over all non-padding tokens of the final hidden state, including the
  two `<dna>` / `</dna>` tag tokens. Embedding dimension 1,024.

## Training data

From the model's own card: 600B 6-mer tokens (~3.6T base pairs), single mixture held
constant for the whole run — 50% Generator-style eukaryotic genes, 25% mature mRNA,
10% splice-enriched mRNA, 15% GTDB bacterial genomes. Cross-entropy loss to 300B tokens,
then Factorised Nucleotide Supervision to 600B. bfloat16, AdamW, no long-context stage.

## Disclosure

- **Zero-shot relative to benchmark tasks:** yes. The encoder is frozen; no GENEB labels
  or task identities reach the model, and nothing beyond the protocol's linear probe is
  trained.
- **Known train/test overlap with benchmark data:** not audited, and plausible. The
  pretraining mixture is eukaryotic genes, mRNA and bacterial genomes; GENEB includes
  human histone-mark, regulatory and open-chromatin tasks drawn from the same reference
  genome, so sequence-level overlap with the benchmark's test splits cannot be ruled out.
  The Carbon cards publish no contamination audit against these task sets.
- **Intended use caveat:** Carbon-500M is published as a *draft model for speculative
  decoding* alongside Carbon-3B and Carbon-8B, which share its tokenizer. Its own card
  states it "is not designed to be competitive with the 3B/8B Carbon models on downstream
  benchmarks." Read this row as the small end of the Carbon family, not as the family's
  best showing.
- **Numerical precision:** weights cast to bfloat16 on GPU, float32 on CPU. Against a
  float32 reference the pooled embeddings differ by ~0.3% of their norm, cosine
  similarity 0.999998. Batch composition changes results only through that same
  bfloat16 rounding: in float32 the extractor is batch-invariant to 5.6e-05 absolute,
  and a short sequence embedded alone versus right-padded against a 4,000 nt partner
  agrees to 3.8e-06.
- **Evaluation environment:** Intel Xeon Platinum 8559C, 16 cores; NVIDIA RTX PRO 6000
  Blackwell Server Edition. Python 3.10.12, scikit-learn 1.7.2, numpy 2.2.6, scipy 1.15.3
  (OpenBLAS 0.3.28/0.3.29), torch 2.11.0+cu128, transformers 5.20.2. Model
  revision `2ac078d4dd6c79b5c1255b24b5c64a1982d97c74`. Harness run with the spec settings
  unchanged (`n_jobs=32`). With `n_jobs=32` the probe fit runs in a loky worker capped at
  `16 // 32` -> 1 BLAS thread, so these numbers were produced single-threaded.
- **Reproducibility caveat:** every `full`-regime probe fit stopped at `max_iter=1000`
  without converging, so scores are sensitive to BLAS summation order (darlednik/GENEB#3).
  Under the spec's `n_jobs=32`, any host with fewer than 64 cores also runs the probe on
  1 thread and should reproduce these numbers with OpenBLAS; on H3K14ac, 8 or 16 threads
  instead moved MCC by up to 0.0042.
- **Harness version:** GENEB-0.1.0
- **Submitted by:** J-ratio (https://github.com/J-ratio)

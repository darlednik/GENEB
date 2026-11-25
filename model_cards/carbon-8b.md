# carbon-8b

- **Display name:** Carbon 8B
- **Parameters:** 8,255,705,088 (checkpoint total; input and output embeddings are untied,
  and the output head is not loaded by the extractor)
- **Architecture:** decoder-only autoregressive transformer, Llama-style
  (`LlamaForCausalLM`), 32 layers, hidden size 4,096, GQA (8 KV heads), RoPE, native
  context 32,768 tokens
- **Weights / URL:** https://huggingface.co/HuggingFaceBio/Carbon-8B
- **Tokenizer / input:** `HybridDNATokenizer` — non-overlapping 6-mers for DNA plus the
  Qwen3 BPE vocabulary for text, 155,776 entries. DNA must be wrapped in `<dna>...</dna>`
  or it falls back silently to BPE; the extractor passes `auto_dna_tags=True` and asserts
  the resulting token count on a probe sequence before embedding anything. One token
  carries 6 bp, so the 32,768-token context covers ~197 kbp and no GENEB task (longest
  10,000 nt, GUE `fungi_species_20`) needs windowing. Any 6-mer containing a non-`ACGT`
  base becomes `<oov>`.
- **Pooling:** mean over all non-padding tokens of the final hidden state, including the
  two `<dna>` / `</dna>` tag tokens. Embedding dimension 4,096.

## Training data

From the model's own card: same pre-training recipe and corpus as Carbon-3B, 1T 6-mer
tokens (~6T base pairs) at sequence length 8,192, with the mixture held fixed through the
decay phase: ~70% Generator-style eukaryotic genomic DNA (with metadata), 16% mRNA, 4%
splice-enriched mRNA, plus bacterial genomes. Cross-entropy loss to 100B tokens, then
Factorised Nucleotide Supervision. A long-context decay stage then extends the native
context from 8,192 to 32,768 tokens. bfloat16, AdamW.

## Disclosure

- **Zero-shot:** no (`zero_shot: false`, as for the other entries). The backbone is
  frozen; only the protocol's linear probe is trained, on each task's training split.
- **Known train/test overlap with benchmark data:** not audited, and plausible. The
  pretraining mixture is eukaryotic genes, mRNA and bacterial genomes; GENEB includes
  human histone-mark, regulatory and open-chromatin tasks drawn from the same reference
  genome, so sequence-level overlap with the benchmark's test splits cannot be ruled out.
  The Carbon cards publish no contamination audit against these task sets.
- **Intended use:** the largest of the Carbon decoder models; Carbon-500M and Carbon-3B
  share its tokenizer, and the same extractor serves all three.
- **Numerical precision:** weights cast to bfloat16 on GPU, float32 on CPU. Against a
  float32 reference on 16 H3K14ac sequences, the pooled embeddings differ by ~0.18% of
  their norm, minimum cosine similarity 0.999993. Batch composition changes results only
  through bfloat16 rounding.
- **Batch size:** 8 (`--batch_size 8`), so that the longest task fit in the GPU memory
  available; Carbon-500M and Carbon-3B used 32.
- **Evaluation environment:** Intel Xeon Platinum 8559C, 16 cores; NVIDIA RTX PRO 6000
  Blackwell Server Edition. Python 3.10.12, scikit-learn 1.7.2, numpy 2.2.6, scipy 1.15.3
  (OpenBLAS 0.3.28/0.3.29), torch 2.11.0+cu128, transformers 5.20.2. Model revision
  `2361b060209e1ccba08691e1dc31a3b8eb9e6816`; the extractor loads model and tokenizer at
  this commit (Carbon's tokenizer code loads its Qwen3 BPE base tokenizer unpinned).
  Harness run with the spec settings unchanged (`n_jobs=32`). With `n_jobs=32` the probe
  fit runs in a loky worker capped at `16 // 32` -> 1 BLAS thread, so these numbers were
  produced single-threaded.
- **Reproducibility caveat:** the `full`-regime probe fits hit `max_iter=1000` without
  converging, so scores are sensitive to BLAS summation order (darlednik/GENEB#3). Under
  the spec's `n_jobs=32`, any host with fewer than 64 cores also runs the probe on 1
  thread and should reproduce these numbers with OpenBLAS; on H3K14ac, 8 or 16 threads
  instead moved MCC by up to 0.0042.
- **Run notes:** run with one to three harness processes sharing the GPU, each scoring
  whole tasks with the unmodified `evaluate_task` from `run_GENEB.py` (verified
  bit-identical to a single `run_GENEB.py` run on six tasks). The run was paused after 33
  tasks and resumed with the same settings; tasks in flight at the pause were redone from
  scratch.
- **Harness version:** GENEB-0.1.0
- **Submitted by:** J-ratio (https://github.com/J-ratio)

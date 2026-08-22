# splicebert-1024nt

- **Display name:** SpliceBERT 1024nt
- **Parameters:** 19,710,474 in the checkpoint
- **Architecture:** Six-layer BERT encoder, 512-dimensional hidden states, 16 attention heads
- **Weights / URL:** [Zenodo record 7995778](https://doi.org/10.5281/zenodo.7995778), `SpliceBERT.1024nt`
- **Checkpoint SHA-256:** `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d` (`pytorch_model.bin`)
- **Tokenizer / input:** Single nucleotides; sequences are uppercased, U is converted to T, and supported IUPAC ambiguity symbols are mapped to N
- **Pooling:** Mean of the final hidden states over non-padding tokens, including `[CLS]` and `[SEP]`

## GENEB inference

Sequences up to 1,024 nt are embedded in one pass. Longer sequences use 1,024-nt windows with a stride of 512 nt. The final window is anchored to the sequence end so every nucleotide is covered, and window embeddings are averaged with equal weight to produce one sequence embedding.

Inference uses PyTorch scaled dot-product attention (SDPA). CUDA devices with BF16 support use BF16 autocast; other CUDA devices and CPU use FP32. The submitted metrics were produced with CUDA BF16. The GENEB probes use the frozen sequence embeddings produced by this procedure.

## Runtime

Install the shared harness requirements and the model-specific dependencies:

```bash
python -m pip install -r harness/requirements.txt
python -m pip install "torch==2.7.1" "transformers==4.53.3"
```

The submitted run used Python 3.11.13, PyTorch 2.7.1+cu128, Transformers 4.53.3, CUDA 12.8, and an NVIDIA RTX 5070 Ti. FP32 fallback requires more GPU memory, so lower `--batch_size` if needed.

## Training data

SpliceBERT was pretrained on more than two million primary RNA sequences from 72 vertebrates. The `SpliceBERT.1024nt` checkpoint was trained on variable-length fragments spanning 64--1,024 nt.

## Disclosure

- **Zero-shot relative to benchmark tasks:** Not claimed; sequence-level overlap has not been established.
- **Known train/test overlap with benchmark data:** Unknown. The vertebrate primary-RNA pretraining corpus may overlap genomic loci represented in GENEB.
- **Short-sequence behavior:** Eleven GENEB tasks include sequences shorter than the 64-nt lower bound used in pretraining. They are evaluated unchanged and should be treated as out of distribution for sequence length.
- **Long-sequence behavior:** Sliding-window inference extends the model beyond its native 1,024-nt context without truncating sequence content.
- **Harness version:** GENEB-0.1.0
- **Submitted by:** Ken Chen

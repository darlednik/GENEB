# GENEB — Genomic Embedding Benchmark (`dev`)

This branch hosts **model-specific development extractors and the local evaluation pipeline** for
[GENEB](https://arxiv.org/abs/2606.04525), a multi-task benchmark for DNA sequence encoders.
The canonical benchmark definition, submission schema, leaderboard, and reference harness are
maintained on [`main`](https://github.com/darlednik/geneb/tree/main).

GENEB comprises **100 classification tasks** across **13 functional categories**. Models are
evaluated using a **linear probe on frozen sequence embeddings** under three regimes — full-data,
10-shot, and 1-shot — with MCC, accuracy, and macro-F1 reported as evaluation metrics.

* **Paper:** https://arxiv.org/abs/2606.04525
* **Benchmark (`main`):** https://github.com/darlednik/geneb
* **Leaderboard:** https://huggingface.co/spaces/darlednik/geneb-leaderboard
* **Task data:** https://huggingface.co/datasets/darlednik/geneb-tasks

On `dev`, model-specific extractors under `embedding_pipeline/extractors/` are maintained prior
to integration into the public reference harness on `main`. For the current minimal reference set,
see `harness/extractors/` and
[`CONTRIBUTING.md`](https://github.com/darlednik/geneb/blob/main/CONTRIBUTING.md) on `main`.

---

## Environment

Install dependencies with [uv](https://docs.astral.sh/uv/) from the pipeline directory:

```bash
cd embedding_pipeline
uv sync
```

---

## Task data

GENEB tasks are distributed as one CSV file per task, with columns `text`, `label`, and `split`
(`train` or `test`). Place the task files in a local directory, for example `GENEB_data/`.

Download the dataset revision pinned in
[`benchmark/benchmark_spec.json`](https://github.com/darlednik/geneb/blob/main/benchmark/benchmark_spec.json)
using the synchronization utility from `main`:

```bash
git checkout main -- tools/sync_geneb_dataset.py benchmark/benchmark_spec.json
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data
```
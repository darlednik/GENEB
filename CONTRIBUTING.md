# Contributing results to GENEB

Paper: https://arxiv.org/abs/2606.04525  
Repository: https://github.com/darlednik/geneb  
Task data: https://huggingface.co/datasets/darlednik/geneb-tasks  
Leaderboard: https://huggingface.co/spaces/darlednik/geneb-leaderboard

To add a model to the GENEB leaderboard, you need to provide three things:

1. an embedding extractor that loads your model and converts DNA sequences into fixed-size
   embeddings;
2. a submission file produced by the GENEB harness;
3. a short model card describing the model and its training data.

Evaluation runs on your hardware. Model weights and intermediate embeddings remain local;
the pull request only needs the extractor code, the final metrics, and the model card.

Externally submitted entries are marked as `self-reported` in submission metadata after review.

---

## Submission workflow

External contributors do not need write access to this repository. Please submit new
models through the standard fork-and-pull-request workflow:

1. Fork `darlednik/geneb`.
2. Create a branch in your fork.
3. Add your extractor, submission file, and model card.
4. Run local validation.
5. Open a pull request to `darlednik/geneb:main`.

Submissions are reviewed for schema completeness, protocol consistency, and plausibility
before being merged into the leaderboard.

---

## Quick start

```bash
# 1. Implement harness/extractors/<module>.py by subclassing BaseEmbeddingExtractor.

# 2. Download the GENEB task data revision specified in benchmark/benchmark_spec.json.
python3 -m pip install "huggingface_hub>=0.24"
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data

# 3. Run the harness locally.
python3 harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --url https://huggingface.co/org/my-model \
  --data_dir ./GENEB_data --device cuda --submitted_by "Your Name"

# 4. Validate the produced submission file.
python3 tools/validate_submission.py submissions/my-model-300m.json
```

Then open a pull request containing:

- `submissions/<model_id>.json`
- `harness/extractors/<module>.py`
- `model_cards/<model_id>.md`

Use `--limit N` to evaluate the first *N* tasks before a full benchmark run.

---

## 1. Extractor

Add one module under `harness/extractors/` that implements `extract_embeddings` and returns
a matrix of shape `(n_sequences, embedding_dim)`:

```python
from .base import BaseEmbeddingExtractor
import numpy as np

class MyModelExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device: str = "cpu"):
        self.name_model = name_model
        self.device = device
        # load model and tokenizer

    def extract_embeddings(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        ...
```

Extractor modules are lightweight wrappers around existing models: they load the encoder,
prepare DNA sequences, compute hidden states, apply the chosen pooling strategy, and return
one embedding vector per sequence. The extractor should not perform task-specific training
or use GENEB labels directly.

---

## 2. Dataset

GENEB uses **100** CSV files named `<task_id>.csv` (see `benchmark/benchmark_spec.json`),
with columns `text`, `label`, and `split` (`train` or `test`).

Download the revision pinned in the spec:

```bash
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data
```

Use the released train/test partitions without relabeling or resplitting. On the Hugging
Face dataset page, each subset corresponds to one task; train and test are separate splits.

---

## 3. Run the harness

```bash
python3 harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --data_dir ./GENEB_data --device cuda
```

For each task, the harness (i) embeds train and test sequences, (ii) fits a logistic
regression probe with the settings in `benchmark_spec.json` (seeds
`[13, 17, 42, 123, 997]`, `max_iter=1000`, BLAS pinned to `blas_threads`), and (iii) reports
mean MCC, accuracy, and macro-F1 over seeds for regimes `full`, `1shot`, and `10shot`.
Output is written to `submissions/<model_id>.json`.

**Legacy workflow:** if you already have per-task `results_*.json` from an earlier pipeline
with the same protocol, convert them with:

```bash
python3 tools/pack_submission.py --raw_dir <results_dir> --model_id <id> ...
```

---

## 4. Model card

Add `model_cards/<model_id>.md` following [model_cards/README.md](model_cards/README.md).
Include architecture, training data, zero-shot status, and any known overlap between
pretraining data and benchmark tasks.

---

## 5. Pull request

A pull request for a new model should include:

- `submissions/<model_id>.json` — metrics produced by the harness;
- `harness/extractors/<module>.py` — code used to compute embeddings;
- `model_cards/<model_id>.md` — model description and training-data notes.

CI validates the submission format and checks that leaderboard generation succeeds. After the pull request is merged into `main`, the public Hugging Face Space is updated
automatically.

---

## What CI verifies

CI checks:

- valid JSON and filename ↔ `model_id` consistency;
- presence of all 100 tasks in each regime (`full`, `1shot`, `10shot`);
- metric values within valid ranges;
- successful leaderboard rebuild.

CI does **not** recompute embeddings, retrain probes, or download third-party model weights.

---

## Protocol summary

- Use the pinned dataset revision and harness settings from `benchmark_spec.json`.
- Do not train on the test split or perform encoder fine-tuning on task labels beyond the
  defined linear probe.
- One submission file per model; one pull request per model is easiest to review.

---

## Contact

Repository and leaderboard contact: [Daria Ledneva](mailto:a.ledn2026@gmail.com).

Questions, suggestions, feedback, and model submissions are welcome.
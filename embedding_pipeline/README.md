Here's a clear and professional `README.md` template for your repository, based on your current usage and structure:

---

## Embedding Evaluation Benchmark

This repository provides a unified pipeline for extracting embeddings from biological sequences using various models (e.g., Enformer) and evaluating them via logistic regression, including few-shot performance.

---

## Directory Structure

```
.
├── data_dir/                # Place your CSV datasets here
│   ├── task1.csv
│   ├── task2.csv
├── extractors/              # Embedding extractor classes (e.g., Enformer, NT, etc.)
├── pipeline/                # Logistic regression and evaluation logic
├── results/                 # Output folder with evaluation metrics
├── main.py                  # Entry point to run evaluation
├── init.py
├── README.md
```

---

## Data Format

Each `.csv` file in `data_dir/` should contain the following columns:

* `text`: input sequence (e.g., ACGT...)
* `label`: integer label (e.g., 0 or 1)
* `split`: either `train` or `test`

Example:

```csv
text,label,split
ACGTACGT...,0,train
GTACGTAC...,1,test
...
```

---

## Run Evaluation

To run embedding extraction and classification on your dataset:

```bash
python main.py \
  --csv_dir data_dir \
  --output_dir results \
  --extractor EnformerPyTorchExtractor \
  --module enformer \
  --device cuda
```

* `--csv_dir`: Path to folder with `.csv` files
* `--output_dir`: Where results (`.json` files) will be saved
* `--extractor`: Class name of the embedding extractor
* `--module`: Module file name (without `.py`) in the `extractors/` folder
* `--device`: Device to run on, e.g. `cuda`, `cuda:1`, or `cpu`

---

## Output

For each task (CSV file), the script saves:

* **baseline** (full logistic regression) metrics
* **few-shot** metrics for `k ∈ {1, 5, 10, 20}`
* Each result is saved in `results/results_<task_name>_<model_name>.json`

Example:

```json
{
  "accuracy": {"mean": 0.91, "std": 0.02},
  "f1_score": {"mean": 0.88, "std": 0.03},
  "mcc": {"mean": 0.85, "std": 0.04}
}
```

---

## Add a New Model

To use a new embedding model:

1. Add your model class to `extractors/your_model.py`
2. Make sure it implements:

```python
extract_embeddings(sequences: List[str], batch_size: int = 8) -> np.ndarray
```

3. Run with:

```bash
python main.py \
  --csv_dir data_dir \
  --output_dir results \
  --extractor YourModelClass \
  --module your_model \
  --device cuda
```
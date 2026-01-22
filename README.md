# DNASurvey
DNA Survey Research

### Synchronize dependences and create enviroment

```bash
uv sync
```

### Options
## Command Line Arguments

### Required Arguments
- `--data_dir` – Path to directory containing .csv files or benchmark folders with training/test data
- `--extractor` – Name of the extractor class to use for embedding generation
- `--name_model` – Path or name of the pre-trained model to load

### Optional Arguments
- `-o, --output_dir` – Directory to save results (default: `results`)
- `--module` – Optional module name for the extractor. Defaults to lowercase of class name
- `-d, --device` – Computation device: `cpu` or `cuda` (default: `cuda`)
- `-b, --batch_size` – Batch size for embedding extraction (default: `4`)
- `--n_jobs` – Number of parallel jobs for logistic regression training (default: `32`)
- `--format_reader` – Input data format: `csv` or `dnalongbench` (optional)
- `--type_train` – Training strategy:
  - `all` – Train on full dataset AND few-shot subsets (1 & 10 examples)
  - `only_few_shot` – Train ONLY on few-shot subsets (1 & 10 examples)
  - `only_full` – Train ONLY on full dataset (default: `all`)

### Example run

```bash
uv run main.py --data_dir ./data_dir/ --output_dir ./results --extractor GenomeOceanExtractor --module genomeocean --device cuda:0 --batch_size 16 --name_model DOEJGI/GenomeOcean-500M --type_train all --format_reader csv --n_jobs 10

```

# DNASurvey
DNA Survey Research


## Run script **run_embed_logreg/enformer/run_enformer.py**:

### Synchronize dependences and create enviroment

```bash
uv sync
```

### Run

```bash
uv run pipeline
```

### Options

- `-b, --batch_size` – batch size (default: 10)  
- `-d, --device` – device: `cpu` or `cuda` (default: cuda)  
- `-o, --output_dir` – path to save outputs (default: .)  
- `--log` – enable logging (INFO)

### Example

```bash
uv sync && uv run pipeline --batch_size=16 --device=cuda --output_dir=. --log

```
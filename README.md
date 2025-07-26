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
<<<<<<< HEAD
uv sync && uv run pipeline --batch_size=16 --device=cuda --output_dir=. --log
```
=======
python run_enformer.py \
  --batch_size 10 \
  --device cuda \
  --output_dir ./results \
  --log
```

## Run script **run_embed_logreg/nucleotide_transformer/run_nt.py**:

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python run_nt.py [options]
```

### Options

- `-b, --batch_size` — batch size (default: 32)  
- `-d, --device` — device (`cpu` or `cuda`, autodetect)  
- `-o, --output_dir` — path to save outputs (default: `.`)  
- `-l, --log` — enable logging (INFO)  
- `--models` — list models (default 4 InstaDeepAI)  

### Example

```bash
python run_nt.py \
  --batch_size 32 \
  --device cuda \
  --output_dir results \
  --log
```
>>>>>>> origin/dev

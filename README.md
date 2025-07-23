# DNASurvey
DNA Survey Research


## Run script **run_embed_logreg/enformer/run_enformer.py**:

### Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_enformer.py [options]
```

### Options

- `-b, --batch_size` – batch size (default: 10)  
- `-d, --device` – device: `cpu` or `cuda` (default: cuda)  
- `-o, --output_dir` – path to save outputs (default: .)  
- `--log` – enable logging (INFO)

## Example

```bash
python run_enformer.py \
  --batch_size 10 \
  --device cuda \
  --output_dir ./results \
  --log
```
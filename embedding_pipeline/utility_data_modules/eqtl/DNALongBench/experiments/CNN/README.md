

# CMP: Contact Map Prediction  
**Supported subsets:**  
- `HFF`  
- `H1hESC`  
- `GM12878`  
- `IMR90`  
- `HCT116`  

```bash
python train.py \
  --task_name contact_map_prediction \
  --root ./data \    # 'Data root directory'
  --save_dir ./results/CMP/GM12878 \
  --subset GM12878 \
  --batch_size 2 \
  --num_epochs 5
```

# ETGP: enhancer_target_gene_prediction
```bash
python train.py \
    --task_name enhancer_target_gene_prediction \
    --root ./data \ 
    --save_dir ./results/ETGP \
    --subset None \
    --batch_size 2
```

# EQTL: eQTL Prediction  
**Supported cell types:**  
- `Adipose_Subcutaneous`  
- `Artery_Tibial`  
- `Cells_Cultured_fibroblasts`  
- `Muscle_Skeletal`  
- `Nerve_Tibial`  
- `Skin_Not_Sun_Exposed_Suprapubic`  
- `Skin_Sun_Exposed_Lower_leg`  
- `Thyroid`  
- `Whole_Blood`  

```bash
python train.py \
  --task_name eqtl_prediction \
  --root ./data \ 
  --save_dir ./results/EQTL/Adipose_Subcutaneous \
  --subset Adipose_Subcutaneous \
  --batch_size 2
```

# RSAP: Regulatory Sequence Activity Prediction
**Supported subsets:**  
- `human`  
- `mouse`  
```bash
python train.py \
    --task_name regulatory_sequence_activity \
    --root ./data \ 
    --save_dir ./results/RSAP/mouse \
    --subset mouse \
    --batch_size 2 \
    --num_epochs 5
```

# TISP: Transcription Initiation Signal Prediction
```bash
python train.py \
    --task_name transcription_initiation_signal_prediction \
    --root ./data \ 
    --save_dir ./results/TISP \
    --subset None \
    --batch_size 2 \
    --num_epochs 5
```


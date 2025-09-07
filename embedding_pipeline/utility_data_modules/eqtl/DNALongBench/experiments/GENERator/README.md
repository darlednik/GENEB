

# CMP: Contact Map Prediction  
**Supported subsets:**  
- `HFF`  
- `H1hESC`  
- `GM12878`  
- `IMR90`  
- `HCT116`  

```bash
python train.py \
  --root ./data \    # 'Data root directory'
  --task_name "contact_map_prediction" \
  --subset "GM12878" \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --save_dir "./results/CMP/GM12878/test" \
  --freeze True 
```

# ETGP: enhancer_target_gene_prediction
```bash
python train.py \
  --root ./data \    # 'Data root directory'
  --task_name "enhancer_target_gene_prediction" \
  --subset "GM12878" \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --save_dir "./results/ETGP/test" \
  --freeze True 
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
  --root ./data \    # 'Data root directory'
  --task_name "eqtl_prediction" \
  --subset "Adipose_Subcutaneous" \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --save_dir "./results/EQTL/AS/test" \
  --freeze True 
```

# RSAP: Regulatory Sequence Activity Prediction
**Supported subsets:**  
- `human`  
- `mouse`  
```bash
python train.py \
  --root ./data \    # 'Data root directory'
  --task_name "regulatory_sequence_activity" \
  --subset "mouse" \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --save_dir "./results/RSAP/mouse/test" \
  --freeze True 
```

# TISP: Transcription Initiation Signal Prediction
```bash
python train.py \
  --root ./data \    # 'Data root directory'
  --task_name "transcription_initiation_signal_prediction" \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --save_dir "./results/TISP/test" \
  --freeze True 
```


from evo2 import Evo2
import argparse 
import csv
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.rsap_dataset import get_dataloader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import math
from scipy.stats import pearsonr

def main():
    """
    Test sequence prediction accuracy using Evo2 models.
    Expected results for forward pass:
    - Evo 2 40B 1m: Loss ~0.216, Accuracy ~91.67%
    - Evo 2 7B 1m: Loss ~0.348, Accuracy ~86.35%
    - Evo 2 1B base: Loss ~0.502, Accuracy ~79.56%
    """
    parser = argparse.ArgumentParser(description="Test Evo2 Model Forward Pass")
    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_7b_base', 'evo2_40b_base', 'evo2_1b_base'], 
                       default='evo2_7b',
                       help="Model to test")
    parser.add_argument("--subset", choices=['human', 'mouse'], 
                       default='mouse',
                       help="Subset")
    
    args = parser.parse_args()

    base_out = f"/ocean/projects/bio240015p/wcheng5/DNALongBench/experiments/Evo2_CMP_ETGP_eQTLP/Evo2_RSAP_TISP/results/RSAP"
    base_out = os.path.join(base_out, args.subset)
    os.makedirs(base_out, exist_ok=True)
    save_path = os.path.join(base_out, "model.pt")
    fw_pred   = open(os.path.join(base_out, "pred.txt"), "w")
    fw_tgt    = open(os.path.join(base_out, "target.txt"), "w")

    
    # Initialize model
    model = Evo2(args.model_name)
    
    _, _, test_loader = get_dataloader("regulatory_sequence_activity", subset = args.subset)
    for batch in test_loader: 
        x, y = batch
        print('x:',len(x[0]))
        print('y:',y.size())
        break
    
    if args.subset == 'mouse':
        task_layer = nn.Sequential(
                nn.Linear(4096, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 896 * 1643), # nn.Linear(64, 896 * 1643) mouse; nn.Linear(64, 896 * 5313) human
            ).to("cuda")
    else:
        task_layer = nn.Sequential(
                nn.Linear(4096, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 896 * 5313), # nn.Linear(64, 896 * 1643) mouse; nn.Linear(64, 896 * 5313) human
            ).to("cuda")
            
    checkpoint = torch.load(save_path, map_location="cpu")
    task_layer.load_state_dict(checkpoint ["state_dict"])
    task_layer.eval()  # or model.train() depending on use

    # all_preds, all_targets = [], []
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader, 1),
                                desc="Eval",
                                unit="step",
                                ncols=80,
                                leave=True):
            # print(batch)
            seq, target = batch
            input_ids = torch.tensor(
                        model.tokenizer.tokenize(seq[0][: 100000]),
                        dtype=torch.int,
                    ).unsqueeze(0).to('cuda:0')
            target = target.to("cuda:0")
                    
            layer_name = 'blocks.28.mlp.l3'

            with autocast(device_type="cuda", dtype=torch.bfloat16):  # or torch.float16 if unsupported
                    _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                    hidden = torch.mean(embeddings[layer_name], dim=1) # [1, 4096]
                    logits = task_layer(hidden.float())   # [1, 2]

            # preds = logits.float().detach().cpu().numpy().squeeze()
            # targs = target.float().detach().cpu().numpy().squeeze()

            # write out
            # fw_pred.write(",".join(map(str, preds)) + "\n")
            # fw_tgt.write("," .join(map(str, targs)) + "\n")

            # all_preds.append(preds)
            # all_targets.append(targs)
            # del embeddings, hidden, logits, input_ids, target, seq

            x = logits.float().detach().cpu().numpy().ravel()
            y = target.float().detach().cpu().numpy().ravel()
            sum_x  += x.sum()
            sum_y  += y.sum()
            sum_x2 += (x * x).sum()
            sum_y2 += (y * y).sum()
            sum_xy += (x * y).sum()
            n      += x.size

            del embeddings, hidden, logits, input_ids, target, seq, x, y
            torch.cuda.empty_cache()
            # break
        
    # fw_pred.close()
    # fw_tgt.close()

    # all_preds = np.array(all_preds) 
    # all_targets = np.array(all_targets)

    # np.save("/ocean/projects/bio240015p/wcheng5/DNALongBench/experiments/Evo2_CMP_ETGP_eQTLP/Evo2_RSAP_TISP/results/RSAP/mouse/all_preds.npy", all_preds)
    # np.save("/ocean/projects/bio240015p/wcheng5/DNALongBench/experiments/Evo2_CMP_ETGP_eQTLP/Evo2_RSAP_TISP/results/RSAP/mouse/all_targets.npy", all_targets)
    # np.save(os.path.join(base_out, "all_preds.npy"),   all_preds)
    # np.save(os.path.join(base_out, "all_targets.npy"), all_targets)

    # r, p = pearsonr(all_preds.flatten(), all_targets.flatten())
    # print(f"Pearson r = {r:.4f} (p = {p:.2e})")

    num = n * sum_xy - sum_x * sum_y
    den = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    r = num / den
    print(f"Pearson r = {r:.4f}")


def evaluate():
    import numpy as np
    from scipy.stats import pearsonr

    # 1) Load the files as 2D arrays (each line → one row of comma‑separated values)
    print('read results')
    
    def load_bracketed(fname):
        rows = []
        with open(fname) as f:
            for line in f:
                # remove leading/trailing whitespace and then the [ ]
                clean = line.strip().lstrip("[").rstrip("]")
                # now the numbers are space-separated
                vals = list(map(float, clean.split()))
                rows.append(vals)
        return np.array(rows)

    targs = load_bracketed(os.path.join(base_out, "target.txt")) 
    print('load targs', targs.shape)

    preds = np.loadtxt(os.path.join(base_out, "pred.txt"), delimiter=",")
    print('load preds', preds.shape)
    
    preds_flat = preds.flatten()
    targs_flat = targs.flatten()

    print('calculating pearson')
    r, p_value = pearsonr(preds_flat, targs_flat)
    print(f"Pearson r = {r:.4f}, p = {p_value:.2e}")

   
    
if __name__ == "__main__":
    main()
    evaluate()
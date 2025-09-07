import argparse
import csv
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evo2 import Evo2

from datasets.enhancer_promoter_dataset import get_dataloader


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
    
    args = parser.parse_args()
    save_path = "evo2/enhancer_promoter/model/model.pt"
    fw_pred = open("evo2/enhancer_promoter/output/pred.txt", "w")
    fw_tgt = open("evo2/enhancer_promoter/output/target.txt", "w")
    
    # Initialize model
    model = Evo2(args.model_name)
    
    test_loader = get_dataloader("data/enhancer_promoter_interaction/CRISPRi_EPI", "test")
    
    task_layer = torch.load(save_path).to("cuda")
    task_layer.eval()  # or model.train() depending on use
    
    for i, batch in enumerate(test_loader):
        seq, target = batch
        input_ids = torch.tensor(
                    model.tokenizer.tokenize(seq[0][: 100000]),
                    dtype=torch.int,
                ).unsqueeze(0).to('cuda:0')
        target = target.to("cuda:0")
                
        layer_name = 'blocks.28.mlp.l3'
        outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
        hidden = torch.mean(embeddings[layer_name], dim=1)
                
        logits = task_layer(hidden.float())   # [1, 2]
        scores = torch.nn.functional.softmax(logits, dim=-1)
        fw_pred.write(str(scores[0][1].cpu().item()) + "\n")
        fw_tgt.write(str(target[0].cpu().item()) + "\n")
        
        fw_pred.flush()
        fw_tgt.flush()
    fw_pred.close()
    fw_tgt.close()
        
    
if __name__ == "__main__":
    main()
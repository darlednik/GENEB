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
    
    # Set random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # Initialize model
    model = Evo2(args.model_name)
    
    train_loader = get_dataloader("data/enhancer_promoter_interaction/CRISPRi_EPI", "train")
    valid_loader = get_dataloader("data/enhancer_promoter_interaction/CRISPRi_EPI", "valid")
        
    task_layer = nn.Linear(4096, 2).to("cuda")
    
    optimizer = torch.optim.Adam(task_layer.parameters(), lr=6e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    val_loss = 10000
    
    for epoch in range(1, 21):
        for i, batch in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            
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
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            
            print("Epoch=%d, Step=%d, loss=%f" % (epoch, i, loss.cpu().item()))

        lr_scheduler.step()
        
        with torch.no_grad():
            this_val_loss = []
            for i, batch in enumerate(valid_loader):
                
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
                loss = F.cross_entropy(logits, target)
                this_val_loss.append(loss.cpu().item())
            this_val_loss_average = np.average(this_val_loss)
            if this_val_loss_average < val_loss:
                val_loss = this_val_loss_average
                torch.save(task_layer, save_path)
            
            print("Epoch=%d, val loss=%f" % (epoch, val_loss))
    
if __name__ == "__main__":
    main()
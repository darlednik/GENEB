import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evo2 import Evo2

from datasets.akita_dataset import get_dataloader


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
    save_path = "evo2/contact_map/HFF/model.pt"
    
    # Set random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # Initialize model
    model = Evo2(args.model_name)
    
    train_loader = get_dataloader("evo2/contact_map/Akita/tfrecords/train-*.tfr", "HFF")
    valid_loader = get_dataloader("evo2/contact_map/Akita/tfrecords/valid-*.tfr", "HFF")
        
    task_layer = nn.Linear(4096*2, 1).to("cuda")
    
    optimizer = torch.optim.Adam(task_layer.parameters(), lr=6e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    val_loss = 10000
    dic = {0: "A", 1: "C", 2: "G", 3: "T"}
    
    for epoch in range(1, 21):
        for i, batch in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            seq, scores = batch
            seq_string = [dic[ind.item()] for ind in seq[0]]
            seq_string = "".join(seq_string)
            
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq_string),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            scores = scores.to("cuda:0")
            
            layer_name = 'blocks.28.mlp.l3'
            outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
            hiddens = embeddings[layer_name]  # [1, 102400, 4096]
            hiddens = torch.mean(hiddens.reshape(hiddens.size(0), -1, 2048, hiddens.size(-1)), dim=2)  # [B, 50, dim]
            norm = torch.sqrt(torch.sum(hiddens * hiddens, dim=-1)).unsqueeze(-1) # [B, L]
            norm = torch.bmm(norm, norm.transpose(1, 2))
            outs = (torch.bmm(hiddens, hiddens.transpose(1, 2))/norm).reshape(hiddens.size(0), -1)
            matrix = hiddens[0]
            vec1 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1).transpose(0, 1)
            vec2 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1)
            vec3 = torch.cat((vec2, vec1), dim=-1).reshape(-1, hiddens.size(-1)*2)
            outs = task_layer(vec3.float()).unsqueeze(0).squeeze(-1) #[1, 50*50, 1]]
            
            loss = F.mse_loss(outs, scores)
            loss.backward()
            optimizer.step()
            
            print("Epoch=%d, Step=%d, loss=%f" % (epoch, i, loss.cpu().item()))
            sys.stdout.flush()

        lr_scheduler.step()
        
        with torch.no_grad():
            this_val_loss = []
            for i, batch in enumerate(valid_loader):
                
                seq, scores = batch
                seq_string = [dic[ind.item()] for ind in seq[0]]
                seq_string = "".join(seq_string)
                
                input_ids = torch.tensor(
                    model.tokenizer.tokenize(seq_string),
                    dtype=torch.int,
                ).unsqueeze(0).to('cuda:0')
                scores = scores.to("cuda:0")
                
                layer_name = 'blocks.28.mlp.l3'
                outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                hiddens = embeddings[layer_name]  # [1, 102400, 4096]
                hiddens = torch.mean(hiddens.reshape(hiddens.size(0), -1, 2048, hiddens.size(-1)), dim=2)  # [B, 50, dim]
                norm = torch.sqrt(torch.sum(hiddens * hiddens, dim=-1)).unsqueeze(-1) # [B, L]
                norm = torch.bmm(norm, norm.transpose(1, 2))
                outs = (torch.bmm(hiddens, hiddens.transpose(1, 2))/norm).reshape(hiddens.size(0), -1)
                matrix = hiddens[0]
                vec1 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1).transpose(0, 1)
                vec2 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1)
                vec3 = torch.cat((vec2, vec1), dim=-1).reshape(-1, hiddens.size(-1)*2)
                outs = task_layer(vec3.float()).unsqueeze(0).squeeze(-1) #[1, 50*50, 1]]
                
                loss = F.mse_loss(outs, scores)
                this_val_loss.append(loss.cpu().item())
            this_val_loss_average = np.average(this_val_loss)
            if this_val_loss_average < val_loss:
                val_loss = this_val_loss_average
                torch.save(task_layer, save_path)
            
            print("Epoch=%d, val loss=%f" % (epoch, val_loss))
            sys.stdout.flush()
    
if __name__ == "__main__":
    main()
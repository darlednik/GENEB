from evo2 import Evo2
import argparse 
import csv
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.tisp_dataset import get_dataloader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

tf.config.set_visible_devices([], "GPU")

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
    save_dir = "/ocean/projects/bio240015p/wcheng5/DNALongBench/experiments/Evo2_CMP_ETGP_eQTLP/Evo2_RSAP_TISP/results/TISP"
    save_path = "/ocean/projects/bio240015p/wcheng5/DNALongBench/experiments/Evo2_CMP_ETGP_eQTLP/Evo2_RSAP_TISP/results/TISP/model.pt"
    
    # Set random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # Initialize model
    model = Evo2(args.model_name)
    print(model)
    
    train_loader, valid_loader = get_dataloader("transcription_initiation_signal_prediction")

    for batch in train_loader: 
        x, y = batch
        print('x:',len(x[0][0]))
        print('y:',y.size())
        break
    
    for batch in valid_loader: 
        x, y = batch
        print('x:',len(x[0]))
        print('y:',y.size())
        break
    
    task_layer = nn.Sequential(
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10 * 100000),
        ).to("cuda")
    
    optimizer = torch.optim.Adam(task_layer.parameters(), lr=6e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = GradScaler()
    val_loss = 10000
    
    for epoch in range(1, 5):
        train_iter = tqdm(enumerate(train_loader, 1), desc=f"Epoch {epoch}", unit="step", ncols=80, leave=True)
        for i, batch in train_iter:
            # zero the parameter gradients
            optimizer.zero_grad()
            
            seq, target = batch
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq[0][0]),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            target = target.to("cuda:0")
  
            layer_name = 'blocks.28.mlp.l3'
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):  
                _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                hidden = torch.mean(embeddings[layer_name], dim=1) 
          
                del embeddings
                torch.cuda.empty_cache()
            hidden.requires_grad_()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = task_layer(hidden.float())
                # print('logits: ', logits.shape)
                loss = F.mse_loss(logits, target.float().view(-1))   
            
            train_iter.set_postfix(loss=f"{loss.cpu().item():.4f}")
            tqdm.write(f"Epoch {epoch} | Step {i} | Loss {loss.cpu().item():.4f}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 1000 == 0:
                fname = os.path.join(
                    save_dir,
                    f"model_epoch{epoch}_step{i}.pt"   
                )
                torch.save({
                    "epoch": epoch,
                    "step": i,
                    "state_dict": task_layer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, fname)
                print(f"→ Saved checkpoint: {fname}")
                
            del hidden, logits, loss
            torch.cuda.empty_cache()




        task_layer.eval()
        lr_scheduler.step()
        
        with torch.no_grad():
            this_val_loss = []
            for i, batch in enumerate(valid_loader):
                
                seq, target = batch
                input_ids = torch.tensor(
                    model.tokenizer.tokenize(seq[0]),
                    dtype=torch.int,
                ).unsqueeze(0).to('cuda:0')
                target = target.to("cuda:0")
                
                layer_name = 'blocks.28.mlp.l3'

                with autocast(device_type="cuda", dtype=torch.bfloat16):  
                    _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                    hidden = torch.mean(embeddings[layer_name], dim=1) 
                    logits = task_layer(hidden.float())  
                    loss = nn.functional.mse_loss(logits, target.float().view(-1))
                    this_val_loss.append(loss.cpu().item())
                    del embeddings, hidden, logits, loss
                    torch.cuda.empty_cache()

            this_val_loss_average = np.average(this_val_loss)
            if this_val_loss_average < val_loss:
                val_loss = this_val_loss_average
                torch.save({
                    "state_dict": task_layer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
            
            print("Epoch=%d, val loss=%f" % (epoch, val_loss))
    
if __name__ == "__main__":
    main()
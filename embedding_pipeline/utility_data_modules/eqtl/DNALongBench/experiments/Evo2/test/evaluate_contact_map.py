import argparse
import csv
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
    for cell in ["HFF", 'H1hESC', 'GM12878', 'IMR90', 'HCT116']:
        print(cell)
        
        save_path = "evo2/contact_map/{}/model.pt".format(cell)
        fw_pred = open("evo2/contact_map/{}/pred.npy".format(cell), "wb")
        fw_tgt = open("evo2/contact_map/{}/target.npy".format(cell), "wb")
        
        # Initialize model
        model = Evo2(args.model_name)
        
        test_loader = get_dataloader("evo2/contact_map/Akita/tfrecords/test-*.tfr", cell)
        
        task_layer = torch.load(save_path).to("cuda")
        task_layer.eval()  # or model.train() depending on use
        dic = {0: "A", 1: "C", 2: "G", 3: "T"}
        prediction = []
        target = []
            
        for i, batch in enumerate(test_loader):
            print(i)
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
            outs = task_layer(vec3.float()).unsqueeze(0).squeeze(-1) #[1, 50*50]]
            
            preds = []
            tgt = []
            for j in range(outs.size(1)):
                preds.append(outs[0][j].item())
                tgt.append(scores[0][j].item())
            prediction.append(preds)
            target.append(tgt)
            
        np.save(fw_pred, np.array(prediction))
        np.save(fw_tgt, np.array(target))
        fw_pred.close()
        fw_tgt.close()
    
if __name__ == "__main__":
    main()
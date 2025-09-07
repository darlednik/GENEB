import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np
import os
from natsort import natsorted
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from dnalongbench import load_data
from task_configs import get_model, get_configs

tf.config.set_visible_devices([], 'GPU')

def trainer(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    save_dir,
    num_epochs=10,
    gradient_accumulation_steps=1,
    evaluation_metric="pcc",
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    best_val = float("-inf") if evaluation_metric != "loss" else float("inf")
    history = {evaluation_metric: []}

    if isinstance(criterion, nn.MSELoss):
        label_dtype = torch.float32
    else:
        label_dtype = torch.long
        
    for epoch in range(1, num_epochs + 1):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch}")):
            if args.task_name == 'eqtl_prediction':
                inputs = batch['x_ref']
                labels = batch['y']
            else:
                inputs, labels = batch
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=label_dtype) #.view(-1) # label_dtype

            outputs = model(inputs).squeeze() #.view(-1)
            
            loss = criterion(outputs, labels) / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        val_res = evaluate_model(
            args, model, val_loader, device, criterion, eval_metrics=evaluation_metric
        )
        score = val_res[evaluation_metric]
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(score)
        else:
            scheduler.step()
        history[evaluation_metric].append(score)

        print(f"Epoch {epoch:2d} — val {evaluation_metric}: {score:.4f}")

        is_improved = (score > best_val) if evaluation_metric != "loss" else (score < best_val)
        if is_improved:
            best_val = score
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f" 💾 new best {evaluation_metric}: {best_val:.4f}")

    print(f"\n✅ Done {num_epochs} epochs. Best {evaluation_metric}: {best_val:.4f}")

    results = {"history": history, "best_val": best_val}

    if test_loader is not None:
        print("\n🧪 Evaluating on test set...")
        test_res = evaluate_model(
            args, model, test_loader, device, criterion, eval_metrics=evaluation_metric
        )
        test_score = test_res[evaluation_metric]
        print(f"Test {evaluation_metric}: {test_score:.4f}")
        results["test_metrics"] = test_res

    print('Training Complete.')

    return results


def evaluate_model(args, model, data_loader, device, criterion, eval_metrics="pcc"):
    if isinstance(eval_metrics, str):
        metrics = [eval_metrics]
    else:
        metrics = list(eval_metrics)

    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    batches = 0
    
    is_regression = isinstance(criterion, nn.MSELoss)
    label_dtype   = torch.float32 if is_regression else torch.long
  
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if args.task_name == 'eqtl_prediction':
                inputs = batch['x_ref']
                labels = batch['y']
            else:
                inputs, labels = batch
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=label_dtype)

            outputs = model(inputs)
     
            if not is_regression: # if outputs.dim() == 2 and outputs.size(1) > 1:
                logits = outputs
                loss   = criterion(logits, labels)
                preds  = torch.argmax(logits, dim=1)
                # for AUROC/AUPRC we need the prob of class ‘1’
                probs  = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
            else:
                loss  = criterion(outputs, labels)
                preds = outputs.flatten()
                labels = labels.flatten()

            print('preds',preds.shape, 'labels', labels.shape)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
         
            batches += 1


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / batches

    results = {"loss": avg_loss}
    for m in metrics:
        if m == "mse":
            results["mse"] = mean_squared_error(all_labels, all_preds)
        elif m == "mae":
            results["mae"] = mean_absolute_error(all_labels, all_preds)
        elif m == "rmse":
            results["rmse"] = np.sqrt(mean_squared_error(all_labels, all_preds))
        elif m == "r2":
            results["r2"] = r2_score(all_labels, all_preds)
        elif m == "pcc":
            print(all_labels.shape, all_preds.shape)
            pcc, _ = pearsonr(all_labels, all_preds)
            results["pcc"] = pcc
        elif m == "auroc":
            results["auroc"] = roc_auc_score(all_labels, all_probs)
        elif m == "auprc":
            results["auprc"] = average_precision_score(all_labels, all_probs)
        else:
            raise ValueError(f"Unsupported metric: {m}")

    return results   




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate DNA LongBench models")
    parser.add_argument('--root', type=str, default='/work/magroup/shared/DNA_LLM/DNALongBench/', help='Data root directory')
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--subset', type=str, default=None, help='Subset name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=None, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--evaluation_metric', type=str, default='pcc', help='Evaluation metric')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and logs')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    model = get_model(args).to(device)
    print(model)

    train_loader, valid_loader, test_loader = load_data(
        root=args.root,
        task_name=args.task_name,
        subset=args.subset,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )

    if args.evaluation_metric is not None:
        evaluation_metric = args.evaluation_metric
        criterion, _ = get_configs(args.task_name)
    else:
        criterion, evaluation_metric = get_configs(args.task_name)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.9, patience=10, threshold=0
    )

    results = trainer(
        args,
        model,
        train_loader,
        valid_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.save_dir,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_metric=evaluation_metric
    )

    print(results)

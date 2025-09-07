import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable, List, Any
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
)
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score
)
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedModel,
    AutoConfig,
)
import dnalongbench
from dnalongbench.utils import load_data, BasenjiDataSet
import tensorflow as tf
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
import math
from typing import Dict
from scipy.stats import t as t_dist
from data_loaders import get_data
from task_configs import get_model #, get_configs
from utils import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], 'GPU')



def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    criterion: torch.nn.Module
) -> Dict[str, float]:
    """
    Streaming evaluation: regression metrics (MSE, MAE, RMSE, R², PCC)
    entirely in PyTorch to avoid NumPy import issues.
    """
    model.eval()

    # Streaming accumulators (all Python floats)
    n = 0
    sum_loss     = 0.0
    sum_sq_err   = 0.0
    sum_abs_err  = 0.0
    sum_pred     = 0.0
    sum_lbl      = 0.0
    sum_pred_sq  = 0.0
    sum_lbl_sq   = 0.0
    sum_pred_lbl = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['y'].view(-1).float().to(device)

            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            preds_t  = outputs['logits'].view(-1)  # might be bfloat16
            preds_f  = preds_t.float()             # cast to float32
            b        = preds_f.size(0)
            n       += b

            # loss
            loss_b = criterion(preds_t, labels).item()
            sum_loss += loss_b * b

            # streaming stats in torch
            diff      = preds_f - labels
            sum_sq_err   += diff.pow(2).sum().item()
            sum_abs_err  += diff.abs().sum().item()
            sum_pred     += preds_f.sum().item()
            sum_lbl      += labels.sum().item()
            sum_pred_sq  += preds_f.pow(2).sum().item()
            sum_lbl_sq   += labels.pow(2).sum().item()
            sum_pred_lbl += (preds_f * labels).sum().item()

    # compute final metrics
    mse  = sum_sq_err / n
    mae  = sum_abs_err / n
    rmse = math.sqrt(mse)

    ss_tot = sum_lbl_sq - (sum_lbl**2) / n
    r2     = 1 - (sum_sq_err / ss_tot) if ss_tot > 0 else float('nan')

    num = n * sum_pred_lbl - sum_pred * sum_lbl
    den = math.sqrt((n * sum_pred_sq - sum_pred**2) * (n * sum_lbl_sq - sum_lbl**2))
    pcc = num / den if den > 0 else 0.0

    if n > 2:
        t_stat = pcc * math.sqrt((n - 2) / (1 - pcc**2))
        p_val  = 2 * (1 - t_dist.cdf(abs(t_stat), df=n-2))
    else:
        p_val = float('nan')

    return {
        'loss':        sum_loss / n,
        'mse':         mse,
        'mae':         mae,
        'rmse':        rmse,
        'r2':          r2,
        'pcc':         pcc,
        'pcc_p_value': p_val,
        'num_samples': n
    }


def train_regression_model(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    save_dir: str = None,
    save_steps: int = 50,
    early_stopping_patience: int = 5,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,

) -> Dict[str, Any]:
    model.train()

    os.makedirs(save_dir, exist_ok=True)
    
    # Use MSE loss for regression
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=1,
        factor=0.5,
        mode='min'
    )


    best_val_loss = float('inf')
    best_val_pcc = -1.0  # Track best PCC as well
   
    global_step = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_pcc': [],
        'val_r2': [],
        'learning_rates': []
    }

    print(f" Starting regression training for {num_epochs} epochs...")
    print(f" Gradient accumulation steps: {gradient_accumulation_steps}")


    start_time = time.time()

    best_train_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        model.train()
        epoch_train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['y'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids,attention_mask=attention_mask)
                logits = outputs['logits']

                if labels.dim() > 1:
                    labels = labels.view(-1).float()

                # MSE loss for regression
                loss = criterion(logits.view(-1), labels)
                loss = loss / gradient_accumulation_steps

                step_train_loss = loss.item() * gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1

            progress_bar.set_postfix({
                'mse_loss': f"{(loss.item() * gradient_accumulation_steps):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

            if global_step > 0 and global_step % save_steps == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{global_step}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step
                }, checkpoint_path)
                print(f" Intermediate checkpoint saved at step {global_step}")
            if step_train_loss < best_train_loss:
                best_train_loss = step_train_loss
                checkpoint_path = os.path.join(save_dir, f'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step
                }, checkpoint_path)
                print(f" [step {global_step}] new best TRAIN loss: {best_train_loss:.4f}")
    
        avg_train_loss = epoch_train_loss / train_steps
        training_history['train_loss'].append(avg_train_loss)
        training_history['learning_rates'].append(scheduler.get_last_lr()[0])

        print(f"\n Evaluating after epoch {epoch + 1}...")
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_pcc'].append(val_metrics['pcc'])
        training_history['val_r2'].append(val_metrics['r2'])

        print(f"\n Epoch {epoch + 1} Summary:")
        print(f"  Train MSE Loss: {avg_train_loss:.4f}")
        print(f"  Val MSE Loss: {val_metrics['loss']:.4f}")
        print(f"  Val PCC: {val_metrics['pcc']:.4f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save model based on lowest validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_pcc = val_metrics['pcc']

            scheduler.step(val_metrics['loss'])

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_pcc': best_val_pcc,
                'global_step': global_step
            }, os.path.join(save_dir, 'best_model.pt'))

            print(f" New best model saved! Val MSE: {best_val_loss:.4f}, Val PCC: {best_val_pcc:.4f}")
       
        epoch_checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_pcc': val_metrics['pcc'],
            'global_step': global_step
        }, epoch_checkpoint_path)
        print(f" Epoch {epoch + 1} checkpoint saved")
        model.train()

    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time/60:.2f} minutes")
    print(f" Best validation MSE loss achieved: {best_val_loss:.4f}")
    print(f" Best validation PCC achieved: {best_val_pcc:.4f}")

    final_metrics = {
        'training_history': training_history, 
        'best_val_loss': best_val_loss,
        'best_val_pcc': best_val_pcc
    }
    
    if test_loader is not None:
        print("\n Evaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device, criterion)
        final_metrics['test_metrics'] = test_metrics

        print(" Final Test Metrics:")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  PCC: {test_metrics['pcc']:.4f}")
        print(f"  PCC p-value: {test_metrics['pcc_p_value']:.6f}")

    return final_metrics




def train_classification_model(
    model:       torch.nn.Module,
    train_loader,
    val_loader,
    test_loader=None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay:  float = 0.01,
    warmup_steps:  int = 0,
    max_grad_norm: float = 1.0,
    save_dir: str = None,
    eval_steps:   int = 50,
    device:       str = "cuda",
    gradient_accumulation_steps: int = 1,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0

    best_auroc = 0.0
    history = {
        'train_loss': [], 'val_loss_steps': [], 'val_auroc_steps': [],
        'epoch_val_loss': [], 'epoch_val_auroc': [], 'learning_rates': []
    }


    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        for step, batch in enumerate(tqdm(train_loader, desc="train"), start=1):
            if args.task_name == 'eqtl_prediction':
                x_alt = batch['x_alt_input_ids'].to(device)
                x_ref = batch['x_ref_input_ids'].to(device)
                labels = batch['y'].view(-1).long().to(device)
                with torch.cuda.amp.autocast():
                    logits = model(x_alt, x_ref)['logits']
                    loss   = criterion(logits, labels) / gradient_accumulation_steps

            else:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['y'].long().to(device)
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
                    loss   = criterion(logits, labels) / gradient_accumulation_steps

            loss.backward()
            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ——— Step‐level eval & checkpoint
                if eval_steps and global_step % eval_steps == 0:
                    print(f"\n Step {global_step} eval…")
                    vm = evaluate_classification_model(model, val_loader, device)
                    loss_s, auroc_s = vm['loss'], vm['auroc']
                    history['val_loss_steps'].append(loss_s)
                    history['val_auroc_steps'].append(auroc_s)
                    print(f"  AUROC {auroc_s:.4f} | Loss {loss_s:.4f}")

                    # save regular checkpoint
                    ckpt_path = os.path.join(save_dir, f"checkpoint-step-{global_step}.pt")
                    torch.save({
                        "epoch": epoch,
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                    }, ckpt_path)
                    print(f" Saved checkpoint: {ckpt_path}")

                    # update best
                    if auroc_s > best_auroc:
                        best_auroc = auroc_s
                        best_path = os.path.join(save_dir, "best_model.pt")
                        torch.save(model.state_dict(), best_path)
                        print(f" New best at step {global_step}: {best_path}")

            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1



        # ——— record train stats
        history['train_loss'].append(epoch_loss / num_batches)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # ——— Epoch‐level eval & checkpoint
        print(f"\n Epoch {epoch+1} eval…")
        vm = evaluate_classification_model(model, val_loader, device)
        loss_e, auroc_e = vm['loss'], vm['auroc']
        history['epoch_val_loss'].append(loss_e)
        history['epoch_val_auroc'].append(auroc_e)
        print(f"  AUROC {auroc_e:.4f} | Loss {loss_e:.4f}")

        if auroc_e > best_auroc:
            best_auroc = auroc_e
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f" New best at epoch {epoch+1}: {best_path}")
        

    elapsed = (time.time() - start_time) / 60
    print(f"\n Done in {elapsed:.2f} min – best AUROC {best_auroc:.4f}")

    results = {'training_history': history, 'best_val_auroc': best_auroc}

    if test_loader is not None:
        print("\n Final test eval…")
        tm = evaluate_classification_model(model, test_loader, device)
        results['test_metrics'] = tm
        print(f"  Test AUROC {tm['auroc']:.4f} | Loss {tm['loss']:.4f}")

    return results


def evaluate_classification_model(model, data_loader, device: str) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []
    batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if args.task_name == 'eqtl_prediction':
                x_alt = batch['x_alt_input_ids'].to(device)
                x_ref = batch['x_ref_input_ids'].to(device)
                labels = batch['y'].view(-1).long().to(device)
                outputs = model(x_alt, x_ref)
                logits = outputs['logits']      # [B, 2]
            else:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['y'].view(-1).long().to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs['logits']

            loss    = loss_fn(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            batches += 1

    avg_loss = total_loss / batches
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'num_samples': len(all_labels)
    }



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
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and logs')
    parser.add_argument('--freeze', type=bool, default=True, help='Whether to freeze model layers')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_loader, valid_loader, test_loader = get_data(args)
    for batch in train_loader: 
        print(batch)
        break

    model = get_model(args)
    model=model.to(torch.bfloat16).to(device)

    print(model)

    print(count_parameters(model))

    if args.freeze:
        # assumes model.base_model exists and has .layers
        for p in model.base_model.parameters():
            p.requires_grad = False
        for layer in model.base_model.layers[-6:]:
            for p in layer.parameters():
                p.requires_grad = True
        print("After freezing:", count_parameters(model))



    if args.task_name == 'enhancer_target_gene_prediction' or args.task_name == 'eqtl_prediction':
        results = train_classification_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_dir=args.save_dir,
        )
        

    else:
        results = train_regression_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_dir=args.save_dir,
        )

    print(results)

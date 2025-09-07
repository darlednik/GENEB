checkpoint = torch.load(f"/work/magroup/wenduoc/DNALongBench/experiments/GENERator/results/RSAP/mouse/v2/best_model.pt", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])

import math
import torch
from tqdm import tqdm
from typing import Dict
from scipy.stats import t as t_dist

def evaluate_model_custom(
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


# Final evaluation on test set if provided
final_metrics = {}

test_loader_custom = DataLoader(
            test_loader.dataset,
            batch_size=4,
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )

criterion = nn.MSELoss()

print("\n🧪 Evaluating on test set...")
test_metrics = evaluate_model_custom(model, test_loader_custom, device, criterion)
final_metrics['test_metrics'] = test_metrics

print("📊 Final Test Metrics:")
for key, value in test_metrics.items():
    print(f"  {key}: {value:.4f}")

print(final_metrics)

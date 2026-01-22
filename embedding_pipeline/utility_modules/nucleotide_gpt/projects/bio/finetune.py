"""
Fine-tuning script for Nucleotide GPT on genomic classification tasks.

This script fine-tunes a pretrained Nucleotide GPT model on downstream genomic
classification tasks from the Genomic Benchmarks dataset.
"""

import argparse
import dataclasses
import io
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import wandb
from google.cloud import storage
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from tqdm import tqdm

from modelling import model


# --- Configuration ---
def get_default_config():
    """Returns default configuration for fine-tuning."""
    return model.Config(
        d_model=2048,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=12,
        key_dim=128,
        vocab_size=6,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=False,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.bfloat16,
        rules=model.mdl_parallel_rules,
        mesh=model.create_mesh(),
        max_lr=2e-5,
        min_lr=2e-5,
        warmup_steps=50,
        total_steps=10000,
        return_sae_intermediates=True,
    )


# --- Tokenization ---
def tokenize_softmasked(sequence):
    """
    Tokenize genomic sequence, treating lowercase and uppercase as the same.
    
    Args:
        sequence: DNA sequence string (can contain uppercase/lowercase ACGTN)
        
    Returns:
        List of token IDs
    """
    VOCAB = ['P', 'A', 'C', 'G', 'T', 'N']
    stoi = {ch: i for i, ch in enumerate(VOCAB)}
    # Map lowercase to same tokens as uppercase
    stoi.update({
        'a': stoi['A'],
        'c': stoi['C'],
        'g': stoi['G'],
        't': stoi['T'],
        'n': stoi['N']
    })
    return [stoi.get(ch.upper() if ch.lower() in 'acgtn' else ch, 0) for ch in sequence]


# --- Data Loading ---
def load_data_from_gcs(bucket_name, dataset_name, split):
    """
    Load genomic benchmark dataset from Google Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        dataset_name: Name of the genomic benchmark dataset
        split: 'train' or 'test'
        
    Returns:
        pandas DataFrame with columns: sequence, label
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"genomic_benchmarks/{dataset_name}_{split}.csv")
    data_string = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data_string.decode('utf-8')))
    return df


def create_sequence_batch(df, sequence_col, batch_size, max_len, tokenize_fn, shuffle=True):
    """
    Create batches of tokenized sequences with padding and segment IDs.
    
    Args:
        df: DataFrame containing sequences and labels
        sequence_col: Name of column containing sequences
        batch_size: Number of sequences per batch
        max_len: Maximum sequence length (will pad/truncate to this)
        tokenize_fn: Function to convert sequence string to token IDs
        shuffle: Whether to shuffle data before batching
        
    Yields:
        Dict with keys: 'x' (tokens), 'segment_ids' (padding mask), 'labels'
    """
    def process_sequence(seq):
        tokens = tokenize_fn(seq[:max_len])
        seg_ids = [1] * len(tokens)
        padding_length = max_len - len(tokens)
        tokens.extend([0] * padding_length)
        seg_ids.extend([0] * padding_length)
        return tokens, seg_ids
    
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    num_samples = len(df)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_sequences = df[sequence_col].iloc[start_idx:end_idx]
        
        batch_tokens = []
        batch_segment_ids = []
        
        for seq in batch_sequences:
            tokens, seg_ids = process_sequence(seq)
            batch_tokens.append(tokens)
            batch_segment_ids.append(seg_ids)
        
        yield {
            'x': np.array(batch_tokens),
            'segment_ids': np.array(batch_segment_ids),
            'labels': np.array(df['label'].iloc[start_idx:end_idx])
        }


# --- Model Functions ---
def input_shardings(mesh, rules):
    """Create input shardings for data parallelism."""
    logical_axes = {
        "x": model.P("batch", "sequence"),
        "segment_ids": model.P("batch", "sequence"),
        "labels": model.P("batch"),
    }
    return jax.tree.map(
        partial(model._logical_to_sharding, mesh=mesh, rules=rules),
        logical_axes
    )


def fwd(x, segment_ids, weights, hidden, predictor, cfg):
    """
    Forward pass for classification.
    
    Extracts the representation at the last non-padding token position
    and applies classification layers.
    """
    _, internals, x = model.forward(x, segment_ids, weights, cfg)
    
    # Find last non-padding position for each sequence
    last_nonzero = jnp.sum(segment_ids > 0, axis=-1)
    indices = last_nonzero[:, None, None] - 1
    
    # Get final token representation
    last_xs = jnp.take_along_axis(x, indices, 1)
    
    # Apply classification layers
    lad = jax.nn.gelu(jnp.einsum("btd,df->btf", last_xs, hidden))
    prediction = jnp.einsum("btf,fv", lad, predictor)[:, 0, :]
    
    return prediction, internals


def compute_loss(weights, x, segment_ids, class_target, cfg):
    """Compute cross-entropy loss and accuracy."""
    weights, hidden, predictor = weights
    prediction, internals = fwd(x, segment_ids, weights, hidden, predictor, cfg)
    ce, acc = model.cross_entropy_loss(
        prediction, 
        class_target, 
        jnp.ones_like(class_target)
    )
    internals['acc'] = acc
    internals['prediction'] = prediction
    return ce, internals


def update_step(weights, x, segment_ids, class_target, opt_state, step, cfg):
    """Single training step with gradient update."""
    (loss, internals), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        weights, x, segment_ids, class_target, cfg
    )
    lr = model.get_lr_with_cosine_decay_and_warmup(
        step, cfg.total_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps
    )
    weights, opt_state, internals = model.update_weights(
        weights, grads, opt_state, lr, step, cfg, internals
    )
    internals["lr"] = lr
    return loss, weights, opt_state, internals


# --- Main Fine-tuning Function ---
def finetune(
    dataset_name,
    bucket_name,
    pretrained_checkpoint_dir,
    finetuned_checkpoint_dir,
    num_classes,
    num_epochs=1,
    batch_size=16,
    max_lr=2e-5,
    min_lr=2e-5,
    use_pretrained=True,
    use_wandb=True,
):
    """
    Fine-tune Nucleotide GPT on a genomic classification task.
    
    Args:
        dataset_name: Name of genomic benchmark dataset
        bucket_name: GCS bucket containing data
        pretrained_checkpoint_dir: Path to pretrained model checkpoint
        finetuned_checkpoint_dir: Path to save fine-tuned checkpoint
        num_classes: Number of output classes
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        use_pretrained: Whether to load pretrained weights (vs random init)
        use_wandb: Whether to log to Weights & Biases
    """
    print(f"Fine-tuning on {dataset_name}")
    
    # Load data
    train_data = load_data_from_gcs(bucket_name, dataset_name, 'train')
    test_data = load_data_from_gcs(bucket_name, dataset_name, 'test')
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Label distribution:\n{train_data['label'].value_counts()}")
    
    # Get max sequence length
    max_seq_len = train_data['sequence'].str.len().max()
    print(f"Maximum sequence length: {max_seq_len}")
    
    # Calculate training steps
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    total_steps = steps_per_epoch * num_epochs
    print(f"Training for {num_epochs} epochs ({steps_per_epoch} steps/epoch)")
    print(f"Total steps: {total_steps}")
    
    # Initialize config
    cfg = get_default_config()
    cfg = dataclasses.replace(
        cfg,
        total_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )
    
    # Setup checkpoint managers
    pretrained_ckpt_manager = model.make_mngr(path=pretrained_checkpoint_dir)
    finetuned_ckpt_manager = model.make_mngr(path=finetuned_checkpoint_dir)
    
    # Load or initialize weights
    if use_pretrained:
        weights, opt_state = model.load(pretrained_ckpt_manager, cfg)
        print('Loaded pretrained checkpoint')
    else:
        weights = model.Weights.init(cfg, jax.random.PRNGKey(42), cfg.mesh, cfg.rules)
        opt_state = model.init_optimizer_state(weights)
        print('Initialized random weights')
    
    # Initialize classification head
    hidden_info = model.TensorInfo(
        jax.ShapeDtypeStruct((cfg.d_model, cfg.d_model), cfg.weight_dtype_at_rest),
        ("d_model", "ffw"),
        jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
    )
    
    predictor_info = model.TensorInfo(
        jax.ShapeDtypeStruct((cfg.d_model, num_classes), cfg.weight_dtype_at_rest),
        ("d_model", "vocab"),
        jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
    )
    
    hidden = hidden_info.initializer(
        jax.random.PRNGKey(42),
        hidden_info.shape.shape,
        hidden_info.shape.dtype
    )
    
    predictor = predictor_info.initializer(
        jax.random.PRNGKey(43),
        predictor_info.shape.shape,
        predictor_info.shape.dtype
    )
    
    # Combine weights
    weights_combined = (weights, hidden, predictor)
    opt_state_combined = (
        opt_state,
        (jnp.zeros_like(hidden), jnp.zeros_like(hidden)),
        (jnp.zeros_like(predictor), jnp.zeros_like(predictor))
    )
    
    # JIT compile functions
    update_step_jit = jax.jit(update_step, static_argnames=['cfg'])
    compute_loss_jit = jax.jit(compute_loss, static_argnames=['cfg'])
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="nucleotide-gpt-finetune",
            name=f"{dataset_name}",
            config={
                "dataset": dataset_name,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "max_seq_len": max_seq_len,
                "use_pretrained": use_pretrained,
                "num_classes": num_classes,
            }
        )
    
    # Training loop
    global_step = 0
    epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
    
    for epoch in epoch_pbar:
        iterator = create_sequence_batch(
            train_data,
            'sequence',
            batch_size=batch_size,
            max_len=max_seq_len,
            tokenize_fn=tokenize_softmasked,
            shuffle=True
        )
        
        step_pbar = tqdm(
            range(steps_per_epoch),
            desc=f'Epoch {epoch+1}',
            leave=False
        )
        
        for i in step_pbar:
            batch = next(iterator)
            batch = jax.device_put(batch, input_shardings(cfg.mesh, cfg.rules))
            
            _, weights_combined, opt_state_combined, internals = update_step_jit(
                weights_combined,
                batch['x'],
                batch['segment_ids'],
                batch['labels'],
                opt_state_combined,
                global_step,
                cfg
            )
            
            accuracy = float(internals["acc"])
            step_pbar.set_postfix({'acc': f'{accuracy:.4f}'})
            
            if use_wandb:
                wandb.log({
                    "train_accuracy": accuracy,
                    "learning_rate": float(internals["lr"]),
                    "step": global_step,
                    "epoch": epoch
                })
            
            global_step += 1
    
    # Save final checkpoint
    print(f"Saving checkpoint after {global_step} steps")
    model.save(
        finetuned_ckpt_manager,
        weights_combined[0],
        opt_state_combined[0],
        step=global_step
    )
    
    # Evaluation
    print('Evaluating on test set...')
    test_iterator = create_sequence_batch(
        test_data,
        'sequence',
        batch_size=32,
        max_len=max_seq_len,
        tokenize_fn=tokenize_softmasked,
        shuffle=False
    )
    
    accs = []
    num_examples = []
    all_preds = []
    all_labels = []
    
    for batch in tqdm(test_iterator, desc='Evaluating'):
        batch = jax.device_put(batch, input_shardings(cfg.mesh, cfg.rules))
        _, internals = compute_loss_jit(
            weights_combined,
            batch['x'],
            batch['segment_ids'],
            batch['labels'],
            cfg
        )
        
        accs.append(internals['acc'])
        num_examples.append(len(batch['labels']))
        predictions = np.argmax(internals['prediction'], axis=1)
        all_preds.extend(predictions)
        all_labels.extend(batch['labels'])
    
    # Calculate metrics
    final_acc = np.sum(np.array(accs) * np.array(num_examples)) / np.sum(num_examples)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    final_mcc = matthews_corrcoef(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\nTest Results:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"MCC: {final_mcc:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    if use_wandb:
        wandb.log({
            "test_accuracy": final_acc,
            "test_mcc": final_mcc,
            "test_f1": final_f1
        })
        wandb.finish()
    
    return weights_combined, opt_state_combined


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Nucleotide GPT on genomic classification tasks'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of genomic benchmark dataset (e.g., human_ensembl_regulatory)'
    )
    parser.add_argument(
        '--bucket_name',
        type=str,
        default='minformer_data',
        help='GCS bucket name containing data'
    )
    parser.add_argument(
        '--pretrained_checkpoint_dir',
        type=str,
        required=True,
        help='GCS path to pretrained model checkpoint'
    )
    parser.add_argument(
        '--finetuned_checkpoint_dir',
        type=str,
        required=True,
        help='GCS path to save fine-tuned checkpoint'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help='Number of output classes for classification'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--max_lr',
        type=float,
        default=2e-5,
        help='Maximum learning rate'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=2e-5,
        help='Minimum learning rate'
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        help='Initialize with random weights instead of pretrained'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    
    finetune(
        dataset_name=args.dataset_name,
        bucket_name=args.bucket_name,
        pretrained_checkpoint_dir=args.pretrained_checkpoint_dir,
        finetuned_checkpoint_dir=args.finetuned_checkpoint_dir,
        num_classes=args.num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        use_pretrained=not args.no_pretrained,
        use_wandb=not args.no_wandb,
    )


if __name__ == '__main__':
    main()

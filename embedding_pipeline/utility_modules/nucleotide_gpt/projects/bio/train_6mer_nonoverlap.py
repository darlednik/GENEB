"""
Training script for NON-OVERLAPPING 6-mer tokenized models with GCS checkpoint saving.

Usage:
python3 train_6mer_nonoverlap.py --checkpoint_dir=gs://minformer_data/checkpoints/6mer_nonoverlap_model --max_seq_len=1536 --vocab_size=4098 --log_every=10
"""

import argparse
import functools
import os
import wandb
from datetime import datetime
from typing import Any

import data_6mer_nonoverlap  # Our new NON-OVERLAPPING 6-mer data loader
import jax
import jax.numpy as jnp
import modelling.model as model
import numpy as np
from jax.profiler import trace
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="DNA NON-OVERLAPPING 6-mer Training Script")
    # Model architecture parameters
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension")
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier")
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    
    # IMPORTANT: Set vocab_size to 4098 for 6-mer
    parser.add_argument("--vocab_size", type=int, default=4098, help="Vocabulary size (4098 for 6-mer)")
    # IMPORTANT: Set max_seq_len to 1536 (divisible by 512) for flash attention compatibility
    parser.add_argument("--max_seq_len", type=int, default=1536, help="Maximum sequence length in tokens (1536 for non-overlapping 6-mer, padded from 1366)")
    
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate model every N steps")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/6mer_nonoverlap", help="Base directory for TensorBoard logs")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="gs://minformer_data/checkpoints/6mer_nonoverlap_model", 
        help="GCS directory for saving checkpoints (e.g., gs://bucket/path)"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint"
    )
    return parser.parse_args()


def clean_key(key):
    cleaned = key.replace("['", "").replace("']", "")
    cleaned = cleaned.replace(".", "/")
    return cleaned


def flatten_pytree(tree):
    leaves = jax.tree_util.tree_map_with_path(lambda p, x: (clean_key(jax.tree_util.keystr(p)), x), tree)
    return jax.tree_util.tree_leaves(leaves, is_leaf=lambda x: isinstance(x, tuple))


def log_metrics(writer, metrics, step):
    flat_metrics = flatten_pytree(metrics)
    for key, value in flat_metrics:
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)


def main():
    args = parse_args()
    
    # Verify vocab size for 6-mer
    if args.vocab_size != 4098:
        print(f"WARNING: vocab_size is {args.vocab_size}, but should be 4098 for 6-mer tokenization!")
        print("Setting vocab_size to 4098...")
        args.vocab_size = 4098
    
    # Verify sequence length is divisible by 512 for flash attention
    if args.max_seq_len % 512 != 0:
        new_len = ((args.max_seq_len + 511) // 512) * 512  # Round up to nearest multiple of 512
        print(f"WARNING: max_seq_len {args.max_seq_len} is not divisible by 512 (required for flash attention)")
        print(f"Setting max_seq_len to {new_len}...")
        args.max_seq_len = new_len

    # Verify checkpoint directory is on GCS
    if not args.checkpoint_dir.startswith("gs://"):
        print(f"WARNING: checkpoint_dir should be a GCS path starting with 'gs://'")
        print(f"Current path: {args.checkpoint_dir}")
        print("Example: gs://minformer_data/checkpoints/6mer_nonoverlap_model")
        response = input("Continue with local checkpoint directory? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please specify a GCS checkpoint directory.")
            return

    # Initialize wandb
    wandb.init(
        project="dna-transformer-6mer-nonoverlap",
        config={
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "query_heads": args.query_heads,
            "key_heads": args.key_heads,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.total_steps,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "vocab_size": args.vocab_size,
            "tokenization": "6-mer non-overlapping",
            "actual_tokens": 1366,  # Actual tokens before padding
            "checkpoint_dir": args.checkpoint_dir
        }
    )

    # Create a unique log directory name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"6mer_nonoverlap_d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

    # Data loading setup for NON-OVERLAPPING 6-mer tokenized human softmasked genome
    print("Loading NON-OVERLAPPING 6-mer tokenized data from GCS...")
    print("Data path: gs://minformer_data/human_softmasked_6mer_nonoverlap/tfrecords/")
    
    iter = data_6mer_nonoverlap.create_iterator(
        file_pattern="gs://minformer_data/human_softmasked_6mer_nonoverlap/tfrecords//record_*.tfrecord",
        batch_size=args.batch_size,
        shuffle=True
    )
    process_batch = data_6mer_nonoverlap.process_batch_6mer_nonoverlap

    # Model configuration
    cfg = model.Config(
        d_model=args.d_model,
        ffw_multiplier=args.ffw_multiplier,
        query_heads=args.query_heads,
        key_heads=args.key_heads,
        num_layers=args.num_layers,
        key_dim=args.key_dim,
        vocab_size=args.vocab_size,  # 4098 for 6-mer
        max_seq_len=args.max_seq_len,  # 1536 (padded from 1366 for flash attention)
        causal=True,
        use_attn_kernel=True,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.bfloat16,
        rules=model.fsdp_rules,
        mesh=model.create_mesh(),
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )

    # Print the config to verify
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"Tokenization: 6-mer NON-OVERLAPPING")
    print(f"Vocab size: {cfg.vocab_size} tokens")
    print(f"Actual sequence length: 1366 tokens (8192 bp / 6)")
    print(f"Padded sequence length: {cfg.max_seq_len} tokens (for flash attention)")
    print(f"Model dimension: {cfg.d_model}")
    print(f"Number of layers: {cfg.num_layers}")
    print(f"Number of attention heads: {cfg.query_heads}")
    print(f"Learning rate: {cfg.max_lr} (max) to {cfg.min_lr} (min)")
    print(f"Batch size: {args.batch_size}")
    print(f"Total training steps: {cfg.total_steps}")
    print(f"Checkpoint directory (GCS): {args.checkpoint_dir}")
    print(f"Checkpoint interval: every {args.checkpoint_interval} steps")
    print("="*60 + "\n")

    # Checkpoint manager setup - this will work with GCS paths
    print(f"Setting up checkpoint manager at: {args.checkpoint_dir}")
    ckpt_manager = model.make_mngr(path=args.checkpoint_dir)

    # Initialize or load weights and optimizer state
    if args.resume_from_checkpoint:
        print(f"Attempting to resume from checkpoint in {args.checkpoint_dir}...")
        try:
            weights, opt_state = model.load(ckpt_manager, cfg)
            start_step = ckpt_manager.latest_step()
            print(f"Successfully resumed from step {start_step}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting from scratch...")
            weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
            opt_state = model.init_optimizer_state(weights)
            start_step = 0
    else:
        print("Initializing new weights for NON-OVERLAPPING 6-mer model...")
        weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
        opt_state = model.init_optimizer_state(weights)
        start_step = 0

    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames="cfg")
    step = functools.partial(step, cfg=cfg)

    print(f"\nStarting training from step {start_step}...")
    print("="*60 + "\n")

    # Training loop
    with SummaryWriter(log_dir) as writer:
        # Log hyperparameters
        writer.add_hparams(
            {
                "d_model": cfg.d_model,
                "num_layers": cfg.num_layers,
                "query_heads": cfg.query_heads,
                "key_heads": cfg.key_heads,
                "max_lr": cfg.max_lr,
                "min_lr": cfg.min_lr,
                "warmup_steps": cfg.warmup_steps,
                "total_steps": cfg.total_steps,
                "batch_size": args.batch_size,
                "max_seq_len": cfg.max_seq_len,
                "vocab_size": cfg.vocab_size,
                "tokenization": "6-mer-nonoverlap",
            },
            {},
        )

        for i in range(start_step, cfg.total_steps):
            next_batch = next(iter)
            # Pass the padded max_seq_len to the batch processor
            batch = process_batch(next_batch, cfg, step_idx=i, max_seq_len=args.max_seq_len)
            batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))

            # Profile first step
            if i == 0:
                with trace(log_dir):
                    loss, weights, opt_state, internals = step(
                        weights,
                        batch["x"],
                        batch["segment_ids"],
                        batch["y"],
                        opt_state,
                        i,
                        aux=batch["aux"],
                    )
                    jax.block_until_ready(loss)
            else:
                loss, weights, opt_state, internals = step(
                    weights, batch["x"], batch["segment_ids"], batch["y"], opt_state, i, aux=batch["aux"]
                )

            if i % args.log_every == 0:
                # Log metrics
                writer.add_scalar("loss", loss, i)
                writer.add_scalar("accuracy", internals["accuracy"], i)
                writer.add_scalar("num_tokens_per_batch", np.sum(batch["segment_ids"] != 0), i)
                log_metrics(writer, internals, i)

                # Log to wandb
                metrics = {
                    "loss": float(loss),
                    "accuracy": float(internals["accuracy"]),
                    "num_tokens_per_batch": int(np.sum(batch["segment_ids"] != 0)),
                    "learning_rate": float(internals["lr"]),
                    "step": i,
                }

                # Add lowercase/uppercase accuracy if available
                if "lowercase_accuracy" in internals:
                    metrics["lowercase_accuracy"] = float(internals["lowercase_accuracy"])
                    metrics["uppercase_accuracy"] = float(internals["uppercase_accuracy"])
                    print(f"Step {i:6d} | Loss: {loss:.4f} | Acc: {internals['accuracy']:.3f} | "
                          f"LC_Acc: {internals.get('lowercase_accuracy', 0):.3f} | "
                          f"UC_Acc: {internals.get('uppercase_accuracy', 0):.3f} | "
                          f"LR: {internals['lr']:.2e}")
                else:
                    print(f"Step {i:6d} | Loss: {loss:.4f} | Acc: {internals['accuracy']:.3f} | "
                          f"LR: {internals['lr']:.2e}")

                wandb.log(metrics)

            # Token distribution logging (less frequent for 6-mer due to larger vocab)
            if i % 5000 == 0 and i > 0:  # Every 5000 steps
                logits, internals_forward, _ = model.forward(
                    batch["x"], 
                    batch["segment_ids"], 
                    weights, 
                    cfg, 
                    cache=None
                )
                
                print("\n" + "-"*60)
                print("Token distribution analysis (NON-OVERLAPPING 6-mer):")
                pred_logits = np.array(logits[0])
                pred_tokens = [int(x) for x in np.argmax(pred_logits, axis=-1)]
                true_tokens = [int(x) for x in np.array(batch["y"][0])]

                # Show distribution summary
                pred_unique, pred_counts = np.unique(pred_tokens, return_counts=True)
                true_unique, true_counts = np.unique(true_tokens, return_counts=True)
                
                print(f"Predicted: {len(pred_unique)} unique tokens")
                print(f"True: {len(true_unique)} unique tokens")
                
                # Check special tokens
                padding_pred = np.sum(np.array(pred_tokens) == 0)
                padding_true = np.sum(np.array(true_tokens) == 0)
                unknown_pred = np.sum(np.array(pred_tokens) == 1)
                unknown_true = np.sum(np.array(true_tokens) == 1)
                
                print(f"Special tokens in predictions - Padding: {padding_pred}, Unknown: {unknown_pred}")
                print(f"Special tokens in true labels - Padding: {padding_true}, Unknown: {unknown_true}")
                print("-"*60 + "\n")

            # Save checkpoint to GCS
            if i > 0 and i % args.checkpoint_interval == 0:
                print(f"\nSaving checkpoint to GCS at step {i}...")
                print(f"Location: {args.checkpoint_dir}")
                try:
                    model.save(ckpt_manager, weights, opt_state, i)
                    print(f"✓ Checkpoint saved successfully at step {i}")
                except Exception as e:
                    print(f"✗ Error saving checkpoint: {e}")
                    print("Continuing training despite checkpoint save failure...")
                print()

    wandb.finish()
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Final step: {cfg.total_steps}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

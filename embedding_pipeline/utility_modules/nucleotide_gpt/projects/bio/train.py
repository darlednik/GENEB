"""
Training script for NucleotideGPT with repetitive  element weighting.

This script trains the model on softmasked human genome data where:
- Lowercase positions indicate repetitive elements
- Uppercase positions indicate unique sequences
- Loss weighting can be adjusted in model.py

Usage: python train.py --batch_size 16 --checkpoint_dir /path/to/checkpoints
"""

import argparse
import functools
import os
import wandb
from datetime import datetime

import data_softmasked
import jax
import jax.numpy as jnp
import modelling.model as model
import numpy as np
from jax.profiler import trace
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="NucleotideGPT Training Script")
    
    # Model architecture parameters
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension") # Model dimension
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier") # Feed-forward network size multiplier
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads") # Number of attention heads
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads") 
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers") # Number of transformer layers
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--vocab_size", type=int, default=6, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=16384, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/nucleotide_gpt", help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/nucleotide_gpt_checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint")
    
    # Data configuration - softmasked human genome
    parser.add_argument("--data_path", type=str, 
                       default="gs://minformer_data/human_softmasked/tfrecords/record_*.tfrecord",
                       help="Path to softmasked human genome TFRecord files")
    
    return parser.parse_args()


def clean_key(key):
    """Clean key for logging."""
    cleaned = key.replace("['", "").replace("']", "")
    cleaned = cleaned.replace(".", "/")
    return cleaned


def flatten_pytree(tree):
    """Flatten a pytree for logging."""
    leaves = jax.tree_util.tree_map_with_path(lambda p, x: (clean_key(jax.tree_util.keystr(p)), x), tree)
    return jax.tree_util.tree_leaves(leaves, is_leaf=lambda x: isinstance(x, tuple))


def log_metrics(writer, metrics, step):
    """Log metrics to TensorBoard."""
    flat_metrics = flatten_pytree(metrics)
    for key, value in flat_metrics:
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)


def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project="nucleotide-gpt",
        config={
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "query_heads": args.query_heads,
            "key_heads": args.key_heads,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.batch_size,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "dataset": "human_softmasked"
        }
    )

    # Create a unique log directory name with key configuration parameters
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

    # Data loading - softmasked human genome
    print("Loading softmasked human genome data...")
    iter = data_softmasked.create_iterator(
        file_pattern=args.data_path,
        batch_size=args.batch_size,
        shuffle=True
    )
    process_batch = data_softmasked.process_batch_softmasked
    

    # Model configuration
    cfg = model.Config(
        d_model=args.d_model,
        ffw_multiplier=args.ffw_multiplier,
        query_heads=args.query_heads,
        key_heads=args.key_heads,
        num_layers=args.num_layers,
        key_dim=args.key_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
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

    # Print configuration
    print("Model Configuration:")
    for field in cfg.__dataclass_fields__:
        print(f"{field}: {getattr(cfg, field)}")

    # Checkpoint manager setup
    ckpt_manager = model.make_mngr(path=args.checkpoint_dir)

    # Initialize or load weights and optimizer state
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        weights, opt_state = model.load(ckpt_manager, cfg)
        start_step = ckpt_manager.latest_step()
        print(f"Resumed from step {start_step}")
    else:
        print("Initializing new weights...")
        weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
        opt_state = model.init_optimizer_state(weights)
        start_step = 0

    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames="cfg")
    step = functools.partial(step, cfg=cfg)

    # Training loop
    print(f"Starting training from step {start_step} to {cfg.total_steps}")
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
            },
            {},
        )

        for i in range(start_step, cfg.total_steps):
            # Get batch and process it
            next_batch = next(iter) 
            batch = process_batch(next_batch, cfg, step_idx=i)
            batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))

            # Profile first step for optimization insights
            if i == 0:
                with trace(log_dir):
                    loss, weights, opt_state, internals = step(
                        weights,
                        batch["x"],
                        batch["segment_ids"],
                        batch["y"],
                        opt_state,
                        i,
                        aux=batch["aux"], # Contains lowercase_mask
                    )
                    jax.block_until_ready(loss)
            else:
                loss, weights, opt_state, internals = step(
                    weights, batch["x"], batch["segment_ids"], batch["y"], opt_state, i, aux=batch["aux"]
                )

            # Logging
            if i % args.log_every == 0:
                # Log to TensorBoard
                writer.add_scalar("loss", loss, i)
                writer.add_scalar("accuracy", internals["accuracy"], i)
                writer.add_scalar("num_tokens_per_batch", np.sum(batch["segment_ids"] != 0), i)
                #print(f"Step {i}, Loss: {loss}, Accuracy: {internals['accuracy']}")
                log_metrics(writer, internals, i)

                # Log to wandb
                metrics = {
                    "loss": loss,
                    "accuracy": internals["accuracy"],
                    "num_tokens_per_batch": np.sum(batch["segment_ids"] !=0),
                    "learning_rate": internals["lr"],
                    "step": i,
                }

                # Log grad norms if available
                if "grad_norms" in internals:
                    grad_norms = jax.tree_util.tree_leaves(internals["grad_norms"])
                    metrics["grad_norm_avg"] = np.mean([x.item() for x in grad_norms])
                
                # Add lowercase/uppercase accuracy (specific to RE weighting)
                if "lowercase_accuracy" in internals:
                    metrics["lowercase_accuracy"] = internals["lowercase_accuracy"]
                    metrics["uppercase_accuracy"] = internals["uppercase_accuracy"]
                    print(f"Step {i}, Loss: {loss:.4f}, Acc: {internals['accuracy']:.3f}, "
                        f"LC_Acc: {internals.get('lowercase_accuracy', 0):.3f}, "
                        f"UC_Acc: {internals.get('uppercase_accuracy', 0):.3f}")
                else:
                    print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {internals['accuracy']:.3f}")

                wandb.log(metrics)

            # Checkpointing
            if i > 0 and i % args.checkpoint_interval == 0:
                print(f"Saving checkpoint at step {i}")
                model.save(ckpt_manager, weights, opt_state, i)

    wandb.finish()
    print("Training completed. TensorBoard logs saved in:", log_dir)


if __name__ == "__main__":
    main()


"""NucleotideGPT Model Definition

A decoder-only transformer for genomic sequences with:
- LLaMA-style architecture (RMSNorm, RoPE)
- Repetitive element weighting for loss computation
- SAE intermediate activation support
- Single-nucleotide tokenization

Adapted from Minformer by Sholto Douglas.
"""

import dataclasses
import math
from collections import namedtuple
from dataclasses import field
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import struct
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax import tree_util as jtu

def create_mesh():
    """Create 1D mesh for FSDP (Fully Sharded Data Parallelism)."""
    devices = jax.devices()
    mesh_shape = (len(devices),)
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape, devices), ("x",))
    return mesh


ShardingRules = namedtuple(
    "ShardingRules",
    ["batch", "sequence", "d_model", "query_heads", "key_heads", "key_dim", "ffw", "vocab"],
)

# Define sharding rules for Fully Sharded Data Parallelism (FSDP)
fsdp_rules = ShardingRules(
    batch="x",  # Shard batch dimension
    sequence=None,  # Don't shard sequence dimension
    d_model="x",  # Shard model dimension
    query_heads=None,
    key_heads=None,
    key_dim=None,
    ffw=None,
    vocab=None,
)

# Define sharding rules for model parallelism (kept for compatibility)
mdl_parallel_rules = ShardingRules(
    batch=None,
    sequence=None,
    d_model=None,
    query_heads="x",  # Shard query heads
    key_heads="x",  # Shard key heads
    key_dim=None,
    ffw="x",  # Shard feed-forward layer
    vocab=None,
)


def _logical_to_physical(logical: P, rules: ShardingRules):
    """Converts logical to physical pspec."""
    return P(*(getattr(rules, axis) for axis in logical))


def _logical_to_sharding(logical: P, mesh: jax.sharding.Mesh, rules: ShardingRules):
    """Converts logical to sharding."""
    return jax.sharding.NamedSharding(mesh, _logical_to_physical(logical, rules))


@struct.dataclass
class Config:
    # Model architecture
    d_model: int  # Model dimension/width
    ffw_multiplier: int  # Feedforward layer scaling factor
    query_heads: int  # Number of attention heads for queries
    key_heads: int  # Number of attention heads for keys
    num_layers: int  # Number of transformer layers
    key_dim: int  # Dimension of keys/queries
    vocab_size: int  # Size of vocabulary
    max_seq_len: int  # Maximum sequence length
    
    # Training configuration
    causal: bool = True  # Use causal (unidirectional) attention
    use_attn_kernel: bool = True  # Use optimized flash attention on TPU
    weight_dtype_at_rest: jnp.dtype = jnp.float32  # Storage dtype
    active_weight_dtype: jnp.dtype = jnp.bfloat16  # Computation dtype
    
    # Sharding configuration
    rules: ShardingRules = None  # Sharding rules for distributed training
    mesh: jax.sharding.Mesh = None  # TPU device mesh
    
    # Optimizer configuration
    max_lr: float = 3e-4  # Maximum learning rate
    min_lr: float = 1e-5  # Minimum learning rate
    warmup_steps: int = 50  # Learning rate warmup steps
    total_steps: int = 10000  # Total training steps
    grad_norm_clip: float = 0.1  # Gradient clipping threshold
    
    # SAE support
    return_sae_intermediates: bool = False  # Return intermediate activations for SAE
    # Note: Layer 6 intermediate activations are extracted for SAE training (hardcoded in forward_layer)
    # This avoids JAX tracing issues with integer config values


@struct.dataclass
class TensorInfo:
    shape: jax.ShapeDtypeStruct
    logical_axes: tuple
    initializer: Callable | None = None
    metadata: dict = field(default_factory=dict)


def process_batch(batch, cfg, step_idx: int | None = None):
    """Process a batch of genomic sequences for training.
    
    Creates input-target pairs for next-token prediction by shifting sequences by 1.
    """
    del step_idx
    batch_size = batch["x"].shape[0]
    dummy = np.zeros((batch_size, 1), dtype=jnp.int32)
    
    return {
        "x": np.concatenate([batch["x"][:, :-1], dummy], axis=-1),
        "y": np.concatenate([batch["x"][:, 1:], dummy], axis=-1),
        "segment_ids": np.concatenate([batch["segment_ids"][:, :-1], dummy], axis=-1),
        "aux": batch.get("aux", None),  # Pass through auxiliary data like lowercase masks
    }


@struct.dataclass
class Layer:
    # Attention weights
    q: jax.Array | TensorInfo  # Query projection
    k: jax.Array | TensorInfo  # Key projection
    v: jax.Array | TensorInfo  # Value projection
    proj: jax.Array | TensorInfo  # Output projection
    
    # Feedforward weights
    w1: jax.Array | TensorInfo  # First FFN layer
    w2: jax.Array | TensorInfo  # Second FFN layer
    
    # Layer normalization (RMSNorm) parameters
    attn_in_gamma: jax.Array | TensorInfo  # Pre-attention norm
    attn_out_gamma: jax.Array | TensorInfo  # Post-attention norm
    ff_in_gamma: jax.Array | TensorInfo  # Pre-FFN norm
    ff_out_gamma: jax.Array | TensorInfo  # Post-FFN norm

    @classmethod
    def abstract(cls, cfg: Config):
        """Define layer structure and initialization."""
        return Layer(
            # Attention projections
            q=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.query_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
                ("d_model", "query_heads", "key_dim"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
            ),
            k=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
                ("d_model", "key_heads", "key_dim"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
            ),
            v=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
                ("d_model", "key_heads", "key_dim"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
            ),
            proj=TensorInfo(
                jax.ShapeDtypeStruct((cfg.query_heads, cfg.key_dim, cfg.d_model), cfg.weight_dtype_at_rest),
                ("query_heads", "key_dim", "d_model"),
                jax.nn.initializers.he_normal(in_axis=(0, 1), out_axis=2),
            ),
            # Feedforward weights
            w1=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.d_model * cfg.ffw_multiplier), cfg.weight_dtype_at_rest),
                ("d_model", "ffw"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            w2=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model * cfg.ffw_multiplier, cfg.d_model), cfg.weight_dtype_at_rest),
                ("ffw", "d_model"),
                jax.nn.initializers.he_normal(in_axis=1, out_axis=0),
            ),
            # RMSNorm parameters
            attn_in_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            attn_out_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            ff_in_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            ff_out_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
        )


@struct.dataclass
class Weights:
    layers: list[Layer]  # Transformer layers
    embedding: jax.Array | TensorInfo  # Token embeddings
    vocab_proj: jax.Array | TensorInfo  # Output projection to vocabulary
    gamma_final: jax.Array | TensorInfo  # Final layer norm

    @classmethod
    def abstract(cls, cfg: Config):
        """Define model structure."""
        return Weights(
            layers=[Layer.abstract(cfg) for _ in range(cfg.num_layers)],
            embedding=TensorInfo(
                jax.ShapeDtypeStruct((cfg.vocab_size, cfg.d_model), cfg.weight_dtype_at_rest),
                ("vocab", "d_model"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            vocab_proj=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.vocab_size), cfg.weight_dtype_at_rest),
                ("d_model", "vocab"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            gamma_final=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
        )

    @classmethod
    def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: dict):
        abstract = cls.abstract(cfg)
        return jax.tree.map(
            lambda info: _logical_to_sharding(info.logical_axes, mesh, rules),
            abstract,
            is_leaf=lambda x: isinstance(x, TensorInfo),
        )

    @classmethod
    def init(cls, cfg: Config, key: jax.random.PRNGKey, mesh: jax.sharding.Mesh, 
             rules: dict, use_low_mem_init: bool = True):
        def _init():
            abstract = cls.abstract(cfg)
            num_leaves = len(jax.tree_util.tree_leaves(abstract))
            key_iter = iter(jax.random.split(key, num_leaves))
            return jax.tree.map(
                lambda info: info.initializer(next(key_iter), info.shape.shape, info.shape.dtype),
                abstract,
                is_leaf=lambda x: isinstance(x, TensorInfo),
            )

        if use_low_mem_init:
            _init = jax.jit(_init, out_shardings=cls.shardings(cfg, mesh, rules))
        return jax.device_put(_init(), cls.shardings(cfg, mesh, rules))


def segment_ids_to_positions(segment_ids):
    """Convert segment IDs to position indices within each segment."""
    def scan_fun(a, b):
        # a[0] is position count, a[1] is previous segment ID
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])
    
    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")


def _generate_pos_embeddings(positions: jax.Array, features: int, 
                            min_timescale=1.0, max_timescale=16384.0):
    """Generate sin/cos for Rotary Position Embeddings (RoPE)."""
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    
    # High precision for position encoding
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(x, sin, cos):
    """Apply rotary position embeddings to queries/keys."""
    assert x.ndim == 4
    assert sin.ndim == 3 and cos.ndim == 3
    x1, x2 = jnp.split(x, 2, axis=-1)
    sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal):
    """Create causal attention mask with segment boundaries."""
    # Segment mask: attend only within same segment
    segment_mask = q_segment_ids[:, :, None] == k_segment_ids[:, None, :]
    segment_mask = segment_mask[:, None, :, :]  # Add head dimension
    
    if causal:
        # Causal mask: attend only to previous positions
        qk = (1, 1, q_len, k_len)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2)
        k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3)
        q_positions = q_iota + q_offset[:, None, None, None]
        causal_mask = q_positions >= k_iota
        combined_mask = jnp.logical_and(segment_mask, causal_mask)
        return combined_mask
    else:
        return segment_mask


def attention(q, k, v, q_segment_ids, k_segment_ids, q_offset, cfg, internals, layer_idx):
    """Multi-head attention computation."""
    scale = q.shape[-1] ** -0.5
    assert q.dtype == jnp.float32
    assert k.dtype == jnp.float32
    qk = jnp.einsum("bhtd,bhTd->bhtT", q, k) * scale
    mask = make_attention_mask(q.shape[2], k.shape[2], q_segment_ids, k_segment_ids, q_offset, cfg.causal)
    qk = jnp.where(mask, qk, -1e30)
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)
    internals['layers'][layer_idx]['attn_scores'] = attn
    return jnp.einsum("bhtT,bhTd->bhtd", attn, v).astype(jnp.bfloat16)


def attention_kernel(q, k, v, q_segment_ids, kv_segment_ids, cfg):
    """Flash attention kernel for TPU."""
    q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)
    scale = q.shape[-1] ** -0.5

    @partial(
        shard_map,
        mesh=cfg.mesh,
        in_specs=(
            _logical_to_physical(P("batch", "query_heads", "sequence", "key_dim"), cfg.rules),
            _logical_to_physical(P("batch", "key_heads", "sequence", "key_dim"), cfg.rules),
            _logical_to_physical(P("batch", "key_heads", "sequence", "key_dim"), cfg.rules),
            _logical_to_physical(P("batch", "sequence"), cfg.rules),
            _logical_to_physical(P("batch", "sequence"), cfg.rules),
        ),
        out_specs=_logical_to_physical(P("batch", "query_heads", "sequence", "key_dim"), cfg.rules),
        check_rep=False,
    )
    def _f(q, k, v, q_segment_ids, kv_segment_ids):
        segment_ids = flash_attention.SegmentIds(q_segment_ids, kv_segment_ids)
        return flash_attention.flash_attention(
            q, k, v,
            segment_ids=segment_ids,
            causal=True,
            sm_scale=scale,
            block_sizes=flash_attention.BlockSizes(
                block_q=512, block_k_major=512, block_k=512, block_b=1,
                block_q_major_dkv=512, block_k_major_dkv=512,
                block_k_dkv=512, block_q_dkv=512,
                block_k_major_dq=512, block_k_dq=512, block_q_dq=512,
            ),
        )

    return _f(q, k, v, q_segment_ids, kv_segment_ids).astype(jnp.bfloat16)


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    """RMS normalization."""
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32)**2, axis=-1, keepdims=True) + 1e-6)
    return jnp.astype(gamma * x / rms, jnp.bfloat16)


def forward_layer(x, segment_ids, layer, sin, cos, idx, cfg, cache=None, internals=None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Forward pass through a single transformer layer."""
    if internals is not None and 'layers' in internals:
        internals['layers'].append({})
    
    # Cast weights to active dtype
    layer = dataclasses.replace(
        layer,
        q=cfg.active_weight_dtype(layer.q),
        k=cfg.active_weight_dtype(layer.k),
        v=cfg.active_weight_dtype(layer.v),
        w1=cfg.active_weight_dtype(layer.w1),
        w2=cfg.active_weight_dtype(layer.w2),
    )
    
    # Pre-attention RMSNorm
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_in_gamma)
    
    # Multi-head attention
    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum("btd,dhq->bhtq", attn_in, layer.q)
        k = jnp.einsum("btd,dhk->bhtk", attn_in, layer.k)
        v = jnp.einsum("btd,dhv->bhtv", attn_in, layer.v)
    
    # Apply rotary embeddings
    with jax.named_scope("rope"):
        q = apply_rotary_embedding(q, sin, cos)
        k = apply_rotary_embedding(k, sin, cos)
    
    with jax.named_scope("cache_update"):
        if cache is not None:
            cache_k, cache_v = cache.k[idx], cache.v[idx]

            def update(original, update, at):
                # Axis -1 because we are in vmap.
                return jax.lax.dynamic_update_slice_in_dim(original, update, at, axis=cache.time_axis - 1)

            k, v = jax.vmap(update, in_axes=(0, 0, 0))(cache_k, k.astype(cache_k.dtype), cache.lengths), jax.vmap(
                update, in_axes=(0, 0, 0)
            )(cache_v, v.astype(cache_v.dtype), cache.lengths)
            q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
            time_indices = jnp.arange(0, v.shape[cache.time_axis])[None, :]  # [1, T]
            incremental_positions = jnp.sum(segment_ids != 0, axis=-1)  # [B,]
            # I.e. valid below where we've written things [B, T]
            k_segment_ids = jnp.where(time_indices < (cache.lengths + incremental_positions)[:, None], 1, 0)
            # Mask our new k and v so that its very visible and easy to test kv values being entered. Tiny perf hit b/c it is unnecessary.
            k, v = k * k_segment_ids[:, None, :, None], v * k_segment_ids[:, None, :, None]
            q_offset = cache.lengths
        else:
            q_segment_ids = segment_ids
            k_segment_ids = segment_ids
            q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
    
    # Compute attention
    with jax.named_scope("attention"):
        if cfg.use_attn_kernel:
            if cache is not None:
                raise ValueError("Kernel is only for training.")
            attn_out = attention_kernel(q, k, v, q_segment_ids, k_segment_ids, cfg)
        else:
            attn_out = attention(q, k, v, q_segment_ids, k_segment_ids, q_offset, cfg, internals, idx)
    
    # Project and apply post-attention norm
    with jax.named_scope("projection"):
        attn_out = jnp.einsum("bhtq,hqd->btd", attn_out, layer.proj)
    
    with jax.named_scope("attn_residual"):
        attn_out = rms_norm(attn_out, layer.attn_out_gamma)
        x = x + attn_out
    
    # Pre-FFN RMSNorm
    with jax.named_scope("ffn_pre_norm"):
        ff_in = rms_norm(x, layer.ff_in_gamma)
    
    # Feedforward network
    with jax.named_scope("ffw"):
        ff_out = jnp.einsum("btd,df->btf", ff_in, layer.w1)
        ff_out = jax.nn.gelu(ff_out)
        ff_out = jnp.einsum("btf,fd->btd", ff_out, layer.w2)
    
    # Apply post-FFN norm and residual
    with jax.named_scope("ffn_residual"):
        ff_out = rms_norm(ff_out, layer.ff_out_gamma)
        x = x + ff_out
    
    # Store intermediate activations for SAE if requested
    if cfg.return_sae_intermediates and idx == 6:  # Layer 6 hardcoded as in original
        internals[f'layer_{idx}_activations'] = x
    
    return x, k, v


def forward(x, segment_ids, weights, cfg, cache=None, aux=None):
    """Forward pass through the full model."""
    internals = {'layers': []} if not cfg.use_attn_kernel else {}
    
    # Token embeddings
    embeds = weights.embedding[x, :]
    x = embeds
    batch = x.shape[0]
    
    # Position embeddings
    positions = segment_ids_to_positions(segment_ids)
    sin, cos = _generate_pos_embeddings(positions, cfg.key_dim, max_timescale=cfg.max_seq_len)
    
    # Forward through layers
    for idx, layer in enumerate(weights.layers):
        x, k, v = forward_layer(x, segment_ids, layer, sin, cos, idx, cfg, cache, internals)
    
    # Final RMSNorm
    x = rms_norm(x, weights.gamma_final)
    
    # Project to vocabulary
    logits = jnp.einsum("btd,dv->btv", x, weights.vocab_proj)
    
    return logits, internals, x


def compute_loss(weights, x, segment_ids, y, cfg, aux=None):
    """Compute loss with optional repetitive element weighting."""
    logits, internals, _ = forward(x, segment_ids, weights, cfg, aux=aux)
    
    # Base mask: exclude padding (segment_id = 0)
    loss_mask = jnp.where(segment_ids == 0, 0, 1)
    
    # Apply repetitive element weighting if lowercase mask is provided
    if aux is not None and "lowercase_mask" in aux:
        lowercase_mask = aux["lowercase_mask"]
        # Weight: 0.0 for lowercase (repetitive elements), 1.0 for uppercase
        # Note: For paper experiments, this was manually changed to 0.0, 0.1, 0.5, or 1.0 to test different RE downweighting strategies
        token_weights = jnp.where(lowercase_mask == 1, 0.0, 1.0)
        loss_mask = loss_mask * token_weights
    
    # Compute cross-entropy loss
    num_classes = logits.shape[-1]
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_token_loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)
    
    # Apply weighted mask
    weighted_loss = per_token_loss * loss_mask
    
    # Normalize by total weight
    total_weight = jnp.sum(loss_mask)
    loss = jnp.sum(weighted_loss) / jnp.maximum(total_weight, 1.0)
    
    # Calculate accuracy (unweighted for true performance)
    predictions = jnp.argmax(logits, axis=-1)
    valid_mask = jnp.where(segment_ids == 0, 0, 1)
    correct = (predictions == y) * valid_mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(valid_mask), 1.0)
    
    internals["token_prediction_loss"] = loss
    internals["accuracy"] = accuracy
    
    # Track separate metrics for repetitive vs non-repetitive regions
    if aux is not None and "lowercase_mask" in aux:
        lowercase_mask = aux["lowercase_mask"]
        # Lowercase (repetitive) accuracy
        lowercase_correct = jnp.sum(correct * lowercase_mask)
        lowercase_total = jnp.maximum(jnp.sum(valid_mask * lowercase_mask), 1.0)
        internals["lowercase_accuracy"] = lowercase_correct / lowercase_total
        
        # Uppercase (non-repetitive) accuracy
        uppercase_mask = (1 - lowercase_mask) * valid_mask
        uppercase_correct = jnp.sum(correct * uppercase_mask)
        uppercase_total = jnp.maximum(jnp.sum(uppercase_mask), 1.0)
        internals["uppercase_accuracy"] = uppercase_correct / uppercase_total
    
    return loss, internals


# Training utilities

def get_lr_with_cosine_decay_and_warmup(step, total_steps, max_lr, min_lr, warmup_steps):
    """Cosine learning rate schedule with linear warmup."""
    def warmup(s):
        return max_lr * (s / warmup_steps)
    
    def cosine_decay(s):
        progress = (s - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress))
    
    return jax.lax.cond(step < warmup_steps, warmup, cosine_decay, step)


def adam_update(param, grad, m, v, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer update."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * jnp.square(grad)
    m_hat = m / (1 - beta1 ** (t + 1))
    v_hat = v / (1 - beta2 ** (t + 1))
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return param - update, m, v


def init_optimizer_state(weights):
    """Initialize Adam optimizer state."""
    def _zeros_like(old):
        if isinstance(old, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(old.shape, old.dtype, sharding=old.sharding)
        else:
            return jax.device_put(jnp.zeros_like(old), old.sharding)
    
    return jtu.tree_map(lambda p: (_zeros_like(p), _zeros_like(p)), weights)


def update_weights(weights, grads, state, lr, t, cfg, internals):
    """Update model weights with gradient clipping."""
    def update_fn(param, grad, state, grad_norm):
        m, v = state
        # Gradient clipping
        scale_factor = jnp.maximum(grad_norm, cfg.grad_norm_clip)
        grad = grad / scale_factor.astype(grad.dtype) * cfg.grad_norm_clip
        param_update, m_new, v_new = adam_update(param, grad, m, v, lr, t)
        return param_update, (m_new, v_new)
    
    grad_norms = jax.tree.map(jnp.linalg.norm, grads)
    internals["grad_norms"] = grad_norms
    updated = jax.tree_map(update_fn, weights, grads, state, grad_norms)
    new_weights = jax.tree.map(lambda _, u: u[0], weights, updated)
    new_state = jax.tree.map(lambda _, u: u[1], weights, updated)
    return new_weights, new_state, internals


def update_step(weights, x, segment_ids, y, opt_state, step, cfg, aux=None):
    """Single training step."""
    (loss, internals), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        weights, x, segment_ids, y, cfg, aux
    )
    lr = get_lr_with_cosine_decay_and_warmup(step, cfg.total_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
    weights, opt_state, internals = update_weights(weights, grads, opt_state, lr, step, cfg, internals)
    internals["lr"] = lr
    return loss, weights, opt_state, internals


def input_shardings(mesh, rules):
    """Define input sharding for distributed training."""
    logical_axes = {
        "x": P("batch", "sequence"),
        "segment_ids": P("batch", "sequence"),
        "y": P("batch", "sequence"),
    }
    physical_axes = jax.tree.map(partial(_logical_to_sharding, mesh=mesh, rules=rules), logical_axes)
    physical_axes["aux"] = None
    return physical_axes


# Checkpointing

def make_mngr(path="/tmp/checkpoint_manager_sharded", erase=False):
    """Create checkpoint manager."""
    if erase:
        path = ocp.test_utils.erase_and_create_empty(path)
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    return ocp.CheckpointManager(path, options=options)


def save(mngr, weights, opt_state, step):
    """Save checkpoint."""
    mngr.save(step, args=ocp.args.StandardSave({"weights": weights, "opt_state": opt_state}))
    mngr.wait_until_finished()


def load(mngr, cfg, step=None):
    """Load checkpoint."""
    import jax
    import orbax.checkpoint as ocp

    abstract_weights = Weights.abstract(cfg)
    single = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def _to_struct(info):
        if info is None:
            return None
        if isinstance(info, TensorInfo):
            return jax.ShapeDtypeStruct(info.shape.shape, info.shape.dtype, sharding=single)
        if isinstance(info, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(info.shape, info.dtype, sharding=single)
        raise TypeError(f"Unexpected leaf type in abstract weights: {type(info)}")

    weights_shapes_shardings = jax.tree.map(
        _to_struct,
        abstract_weights,
        is_leaf=lambda x: (x is None) or isinstance(x, TensorInfo) or isinstance(x, jax.ShapeDtypeStruct),
    )

    restored = mngr.restore(
        mngr.latest_step() if step is None else step,
        args=ocp.args.PyTreeRestore(
            item={"weights": weights_shapes_shardings},
            partial_restore=True,
        ),
    )

    return restored["weights"], None

    #opt_shapes_shardings = init_optimizer_state(weights_shapes_shardings)
    #restored = mngr.restore(
    #    mngr.latest_step() if step is None else step,
    #    args=ocp.args.StandardRestore({"weights": weights_shapes_shardings, "opt_state": opt_shapes_shardings}),
    #    partial_restore=True
    #)
    #return restored["weights"], restored["opt_state"]


# SAE checkpoint support

def make_sae_mngr(path="/tmp/sae_checkpoint_manager", erase=False):
    """Create checkpoint manager for SAE weights."""
    if erase:
        path = ocp.test_utils.erase_and_create_empty(path)
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    return ocp.CheckpointManager(path, options=options)


def save_sae(mngr, sae_weights, sae_opt_state, step):
    """Save SAE checkpoint."""
    to_save = {"sae_weights": sae_weights, "sae_opt_state": sae_opt_state}
    mngr.save(step, args=ocp.args.StandardSave(to_save))
    mngr.wait_until_finished()


def load_sae(mngr, cfg, step=None):
    """Load SAE checkpoint."""
    weights_shapes = {
        'expand': jax.ShapeDtypeStruct((cfg.d_model, 8192), jnp.float32),
        'contract': jax.ShapeDtypeStruct((8192, cfg.d_model), jnp.float32)
    }
    
    opt_shapes = {
        'expand': (
            jax.ShapeDtypeStruct((cfg.d_model, 8192), jnp.float32),
            jax.ShapeDtypeStruct((cfg.d_model, 8192), jnp.float32)
        ),
        'contract': (
            jax.ShapeDtypeStruct((8192, cfg.d_model), jnp.float32),
            jax.ShapeDtypeStruct((8192, cfg.d_model), jnp.float32)
        )
    }
    
    restored = mngr.restore(
        mngr.latest_step() if step is None else step,
        args=ocp.args.StandardRestore({
            "sae_weights": weights_shapes,
            "sae_opt_state": opt_shapes
        })
    )
    
    return restored["sae_weights"], restored["sae_opt_state"]

import os
import traceback, signal
import sys
def catch_signal(sig, frame):
    print(f"Caught signal {sig}, likely external kill.")
    traceback.print_stack()
    sys.exit(1)

def catch_uncaught_exception(exc_type, exc_value, exc_tb):
    print("Uncaught Exception:")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.exit(1)

sys.excepthook = catch_uncaught_exception
signal.signal(signal.SIGTERM, catch_signal)
signal.signal(signal.SIGINT, catch_signal)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import math
import csv
import random
import numpy as np
from tqdm import tqdm
from itertools import islice
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from safetensors.torch import load_file as load_safetensors
from BiMambaForMaskedLM import BiMambaForMaskedLM

import wandb
from transformers.integrations import WandbCallback
from transformers import TrainerCallback
import torch, gc, psutil, os

class MemoryTrackerCallback(TrainerCallback):
    def __init__(self, step_interval=100, log_to_wandb=True):
        self.step_interval = step_interval
        self.log_to_wandb = log_to_wandb
        self.proc = psutil.Process(os.getpid())

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.step_interval != 0:
            return

        cpu_mem_gb = self.proc.memory_info().rss / 1024 ** 3
        gpu_mem_gb = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0

        print(f"[Step {state.global_step}] CPU Mem: {cpu_mem_gb:.2f} GB | GPU Mem: {gpu_mem_gb:.2f} GB")

        if self.log_to_wandb and wandb.run is not None:
            wandb.log({
                "memory/cpu_rss_gb": cpu_mem_gb,
                "memory/gpu_allocated_gb": gpu_mem_gb,
                "global_step": state.global_step
            })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train_model():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.init(project="EccDNA-Foundation-Model", name="Bi-mamba")
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("saved_model/")
    
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.cls_token = "[CLS]"
    
    config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = 0
    config.tie_embeddings = False
    
    # Reconstruct training arguments
    model = BiMambaForMaskedLM(config)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model.to(f"cuda:{local_rank}")
        
    # Data collator for span masking training
    class SpanMaskingCollator:
        """
        SpanBERT-style span masking collator.
        """
        def __init__(self, tokenizer, noise_density=0.15, mean_span_length=3):
            self.tok = tokenizer
            self.noise_density = noise_density
            self.mean_span_len = mean_span_length
            self.special_ids = {tokenizer.cls_token_id,
                                tokenizer.pad_token_id}
                                
        def __call__(self, examples):
            try:
                pairs = [self._mask_span(e["input_ids"]) for e in examples]
            except Exception as e:
                print(" Error in masking spans. Sample batch:", examples)
                raise e
            # pairs = [self._mask_span(e["input_ids"]) for e in examples]
            input_ids, labels = zip(*pairs)

            input_ids = [self.append_head_to_tail(seq, copy_len=64) for seq in input_ids]
            labels    = [self.append_head_to_tail(seq, copy_len=64) for seq in labels]

            max_len = max(map(len, input_ids))

            # pad
            input_ids = torch.stack([self._pad(seq, max_len, self.tok.pad_token_id) for seq in input_ids])
            labels    = torch.stack([self._pad(seq, max_len, -100) for seq in labels])

            return {
                "input_ids":      input_ids,
                "attention_mask": (input_ids != self.tok.pad_token_id).long(),
                "labels":         labels,
            }
    
        def _pad(self, seq, max_len, pad_id):
            return torch.tensor(seq + [pad_id] * (max_len - len(seq)), dtype=torch.long)
    
        def append_head_to_tail(self, seq, copy_len=128):
            copy_len = min(len(seq) - 1, copy_len)
            return seq + seq[1: 1 + copy_len]

        def _mask_span(self, ids):
            ids = list(ids)
            L = len(ids)
            if L <= 2:
                return ids, [-100] * L
            
            k = max(1, int(round(L * self.noise_density)))
    
            candidate = self._sample_spans(ids, L, k)
            labels = [-100] * L
    
            for span in candidate:
                for pos in range(*span):
                    labels[pos] = ids[pos]
                    p = random.random()
                    if p < 0.8:
                        ids[pos] = self.tok.mask_token_id
                    elif p < 0.9:
                        ids[pos] = random.randint(0, self.tok.vocab_size - 1)
                    # else: keep original
    
            return ids, labels
    
        def _sample_spans(self, ids, L, k, max_sampling_attempts=200):
            spans, covered = [], 0
            attempts = 0
            while covered < k and attempts < max_sampling_attempts:
                attempts += 1
                span_len = max(1, int(np.random.poisson(self.mean_span_len)))
                if span_len > L:
                    span_len = L
                start = random.randint(0, L - span_len) if L - span_len > 0 else 0
    
                if any(ids[s] in self.special_ids for s in range(start, start + span_len)):
                    continue
                if any(not (end <= start or start + span_len <= st) for st, end in spans):
                    continue
    
                spans.append((start, start + span_len))
                covered += span_len

            if covered < k:
                print(f"[WARN] span sampling stopped early: L={L}, needed={k}, got={covered}")

            spans.sort()
            return spans
    
    # print("=== CUDA & Precision Check ===")
    # print(f"CUDA Available: {torch.cuda.is_available()}")
    # print(f"Using Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # # Check if bfloat16 is supported
    # device_cap = torch.cuda.get_device_capability()
    # bf16_supported = device_cap >= (8, 0)  # Ampere or newer
    # print(f"BF16 Supported: {bf16_supported} (CUDA Compute Capability: {device_cap})")
    
    # # Check model dtype (will be float32 before AMP is applied)
    # print(f"Model parameter dtype (first layer): {next(model.parameters()).dtype}")
    # print("================================")
    
    
    training_args = TrainingArguments(
        output_dir="./saved_model/",
        per_device_train_batch_size = 6,     
        gradient_accumulation_steps = 8,     
        dataloader_num_workers = 0, 
        # callbacks=[MemoryTrackerCallback(step_interval=500)], 
        learning_rate=5e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        num_train_epochs=3,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_steps=50,                   
        save_steps=5000,                    
        save_total_limit=2,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",  # Enable WandB logging
        run_name="Bi-mamba",           # Same as before
        gradient_checkpointing=True,      # Off to avoid extra memory usage
        ddp_find_unused_parameters=True,
        max_grad_norm=1.0 
    )
    
    tokenized_dataset = load_from_disk("tokenized_dataset/")
    data_collator = SpanMaskingCollator(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_span_length=3
    )
    
    # ── Trainer ───────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[MemoryTrackerCallback(step_interval=10)] 
    )
    
    # Resume training
    try:
        trainer.train()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    trainer.save_model("saved_model/final/")
    print("Model saved to saved_model/final/")

if __name__ == "__main__":
    train_model()

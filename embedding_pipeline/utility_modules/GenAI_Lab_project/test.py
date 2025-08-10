import torch
from transformers import AutoConfig
from BiMambaForMaskedLM import BiMambaForMaskedLM

# Load config
config = AutoConfig.from_pretrained("state-spaces/mamba-130m")
config.vocab_size = 4101  # Make sure this matches your actual vocab size
config.pad_token_id = 0

# Try to initialize the model and move it to GPU
try:
    model = BiMambaForMaskedLM(config)
    model = model.half()  # or .to(torch.bfloat16) if you're using bf16
    model.cuda()
    print("✅ Model loaded successfully on GPU!")
    print(torch.cuda.memory_summary())

except RuntimeError as e:
    print("❌ CUDA RuntimeError:")
    print(e)

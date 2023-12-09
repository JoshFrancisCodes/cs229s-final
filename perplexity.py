import numpy as np
import os
import torch
from contextlib import nullcontext
from model import GPT
from quantize import CustomGPT2Model

batch_size = 4
device_type = 'cuda'
device = 'cuda'
data_dir = "./wikitext"
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
block_size = 1024
eval_iters = 200
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

gpt_model = GPT.from_pretrained('gpt2', dict(dropout=0.2))
custom_model = CustomGPT2Model(gpt_model.config, gpt_model.state_dict())
#model = GPT.from_pretrained('gpt2', dict(dropout=0.2))
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def get_batch():
    data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def perplexity_score(probs: np.ndarray) -> float:
    """Compute perplexity score from softmax probabilities."""
    return np.exp(-np.mean(np.log(probs)))

def get_perplexity():
    # custom_model.to(device)
    #model.to(device)
    #model.eval()
    custom_model.eval()
    custom_model.to(device)

    all_probs = []
    for k in range(eval_iters):
        print(f"Eval iter {k}")
        X, Y = get_batch()
        X.to(device)
        Y.to(device)
        with ctx:
            logits, _ = custom_model.forward(X, Y)
            softmax = torch.softmax(logits, dim=-1)

            for i in range(X.size(0)):  # Iterate over batch
                for j in range(X.size(1) - 1):  # Iterate over sequence length
                    true_next_token = Y[i, j].item()
                    prob = softmax[i, j, true_next_token].item()
                    all_probs.append(prob)

    perplexity = perplexity_score(np.array(all_probs))
    return perplexity

print(get_perplexity())
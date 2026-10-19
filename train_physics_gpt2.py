"""Continue pretraining the TinyStories GPT model on a mixed physics corpus.

This stage starts from tinystories_gpt.pt and mixes three sources in each
training batch:
  - 75% physics question-answer data
  - 15% physics web documents
  - 10% TinyStories replay data

Replay data helps reduce the risk of forgetting general English learned during
initial training. Validation loss is measured separately on physics web text,
physics Q&A, quantum-mechanics Q&A, and TinyStories. The initial model's metrics
provide a baseline for assessing changes after physics training.

Required inputs: tinystories_gpt.pt, train.bin, val.bin, and the prepared
physics .bin files. The model architecture must match the initial checkpoint.
"""
import math
import os
import random
import time

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------
# Model architecture must match Step 3b.
vocab_size = 50304
real_vocab = 50257       # Actual GPT-2 tokenizer vocabulary; the remaining 47 padded IDs must never be generated.
block_size = 256
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1            # Physics data is smaller, so dropout is enabled. Dropout adds no parameters and does not affect checkpoint tensor shapes.

# Training
batch_size = 32
max_iters = 4000
eval_interval = 500
eval_iters = 20
learning_rate = 3e-4     # Use a lower rate than Step 3 to learn physics while reducing catastrophic forgetting.
min_lr = 3e-5
warmup_iters = 100
grad_clip = 1.0

MIX = {"qa": 0.75, "web": 0.15, "stories": 0.10}
TRAIN_FILES = {"qa": "qa_train.bin", "web": "web_train.bin", "stories": "train.bin"}
VAL_FILES = {"qa": "qa_val.bin", "web": "web_val.bin", "qm": "qm_val.bin", "stories": "val.bin"}
INIT_CKPT = "tinystories_gpt.pt"
OUT_CKPT = "physics_gpt.pt"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = device == "cuda"
random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
print("Device:", device)

# ------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------
train_mm = {k: np.memmap(f, dtype=np.uint16, mode="r") for k, f in TRAIN_FILES.items()}
val_mm = {k: np.memmap(f, dtype=np.uint16, mode="r") for k, f in VAL_FILES.items()}
for name, mm in {**{f"train/{k}": v for k, v in train_mm.items()}, **{f"val/{k}": v for k, v in val_mm.items()}}.items():
    print(f"  {name:<14} {len(mm):>12,} tokens")
    assert len(mm) > block_size + 2, f"{name} is too short or empty"


def batch_from(data):
    """Sample batch_size random windows from a data array."""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def get_train_batch():
    """Choose a source for each example according to the MIX probabilities."""
    names = random.choices(list(MIX), weights=list(MIX.values()), k=batch_size)
    xs, ys = [], []
    for name in names:
        data = train_mm[name]
        i = random.randint(0, len(data) - block_size - 2)
        chunk = data[i : i + block_size + 1].astype(np.int64)
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device)


# ------------------------------------------------------------
# 3. Model architecture must match Step 3b.
# ------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Tie the input embedding and output projection weights.

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            logits[:, real_vocab:] = -float("inf")           # Never sample token IDs that are not present in the tiktoken vocabulary.
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


model = GPT().to(device)
model.load_state_dict(torch.load(INIT_CKPT, map_location=device))
print(f"Loaded {INIT_CKPT}. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ------------------------------------------------------------
# 4. Training helpers
# ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
amp_ctx = torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


@torch.no_grad()
def mean_loss(data):
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = batch_from(data)
        with amp_ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean().item()


def evaluate():
    model.eval()
    val = {k: mean_loss(v) for k, v in val_mm.items()}
    train = {k: mean_loss(train_mm[k]) for k in ("qa", "web")}   # Track validation metrics to detect overfitting.
    model.train()
    return val, train


# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
best = float("inf")
t0 = time.time()
for it in range(max_iters + 1):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it)

    if it % eval_interval == 0:
        val, train = evaluate()
        print(f"step {it:>5} | VAL   " + "  ".join(f"{k} {v:.3f}" for k, v in val.items()))
        print(f"           | TRAIN " + "  ".join(f"{k} {v:.3f}" for k, v in train.items()))
        score = val["qa"] + val["web"]
        if it > 0 and score < best:
            best = score
            torch.save(model.state_dict(), OUT_CKPT)

    xb, yb = get_train_batch()
    with amp_ctx:
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    if it % 100 == 0 and it > 0:
        dt = (time.time() - t0) / 100
        t0 = time.time()
        print(f"  iter {it}: loss {loss.item():.4f}, {dt * 1000:.0f} ms/iter")

# ------------------------------------------------------------
# 6. Generate text using the best checkpoint.
# ------------------------------------------------------------
if os.path.exists(OUT_CKPT):
    model.load_state_dict(torch.load(OUT_CKPT, map_location=device))
model.eval()
enc = tiktoken.get_encoding("gpt2")
prompts = [
    "Question: What is the Heisenberg uncertainty principle?\nAnswer:",
    "Question: Why does a spinning ice skater speed up when she pulls her arms in?\nAnswer:",
    "Question: What is the Schrodinger equation used for?\nAnswer:",
    "Once upon a time",   # Check how much story-generation ability has been retained.
]
for p in prompts:
    ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=180)
    text = enc.decode(out[0].tolist()).split("<|endoftext|>")[0]
    print("\n" + "=" * 60)
    print(text)

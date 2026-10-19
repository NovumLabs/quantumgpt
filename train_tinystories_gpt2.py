"""Train a GPT language model on TinyStories using the GPT-2 BPE tokenizer.

Compared with the character-level demonstration, this experiment uses:
  - BPE tokens with a vocabulary of approximately 50,000 entries
  - A larger model with approximately 17 million parameters
  - Tied input-embedding and output-projection weights
  - Mixed precision, a learning-rate schedule, and gradient clipping
  - Best-checkpoint saving and top-k text sampling

Prepare train.bin and val.bin with prepare_tinystories_gpt2_data.py first.
"""
import math
import time

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------
vocab_size = 50304       # GPT-2 has 50,257 tokens; pad to a multiple of 64 for potentially faster GPU execution.
block_size = 256
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.0            # The dataset is relatively large for this small model, so dropout is disabled.

batch_size = 32
max_iters = 5000
eval_interval = 500
eval_iters = 20
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 100
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = device == "cuda"
torch.manual_seed(1337)
print("Device:", device)

# ------------------------------------------------------------
# 2. Read data from disk without loading the full dataset into RAM.
# ------------------------------------------------------------
train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------------
# 3. Model
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")  # Mask all token IDs outside the top-k candidates.
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = GPT().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ------------------------------------------------------------
# 4. Training helpers
# ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
amp_ctx = torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp)


def get_lr(it):
    """Prothome dhire dhire barbe (warmup), tarpor cosine curve-e kombe."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with amp_ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
best_val = float("inf")
t0 = time.time()
for it in range(max_iters + 1):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it)

    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {get_lr(it):.2e}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save(model.state_dict(), "tinystories_gpt.pt")

    xb, yb = get_batch("train")
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
# 6. Generate
# ------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
model.eval()
prompts = ["Once upon a time", "One day, a little girl named Lily", "Tom had a red ball. He"]
for p in prompts:
    ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=150)
    print("\n" + "=" * 60)
    print(enc.decode(out[0].tolist()))

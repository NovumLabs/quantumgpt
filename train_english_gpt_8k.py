"""Train a larger English/TinyStories model from scratch with the 8k tokenizer.

Compared with the earlier GPT-2-tokenizer model, this experiment uses the
custom tokenizer_8k.json vocabulary and increases the embedding dimension,
number of Transformer layers, and context length. The smaller vocabulary reduces
the embedding table, allowing a larger share of parameters to reside in the
Transformer blocks.

Required inputs: tokenizer_8k.json, stories_train_8k.bin, and stories_val_8k.bin.

Run:
    python -m pip install tokenizers
    python train_english_gpt_8k.py
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------
vocab_size = 8192
block_size = 384
n_layer = 8
n_head = 6
n_embd = 384
dropout = 0.0

batch_size = 32
max_iters = 6000
eval_interval = 500
eval_iters = 20
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 150
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = device == "cuda"
torch.manual_seed(1337)
print("Device:", device)

tok = Tokenizer.from_file("tokenizer_8k.json")
EOT = tok.token_to_id("<|endoftext|>")
assert tok.get_vocab_size() <= vocab_size

# ------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------
train_data = np.memmap("stories_train_8k.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("stories_val_8k.bin", dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------------
# 3. Model: same general structure as Step 3b, but with larger dimensions.
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
        self.lm_head.weight = self.tok_emb.weight
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
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
n_emb_params = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
print(f"Total parameters: {n_params / 1e6:.2f}M  (embedding: {n_emb_params / 1e6:.2f}M, "
      f"transformer: {(n_params - n_emb_params) / 1e6:.2f}M)")

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
            torch.save(model.state_dict(), "english_gpt_8k.pt")

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
model.load_state_dict(torch.load("english_gpt_8k.pt", map_location=device))
model.eval()
prompts = ["Once upon a time", "One day, a little girl named Lily", "Tom had a red ball. He"]
for p in prompts:
    ids = torch.tensor([tok.encode(p).ids], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=150)
    text = tok.decode(out[0].tolist())
    print("\n" + "=" * 60)
    print(text)

import os
import shutil

drive_dir = os.environ.get("QM_DRIVE_DIR", "")
if drive_dir and os.path.isdir(os.path.dirname(drive_dir)):
    os.makedirs(drive_dir, exist_ok=True)
    shutil.copy2("english_gpt_8k.pt", os.path.join(drive_dir, "english_gpt_8k.pt"))
    print(f"\nBacked up checkpoint to: {drive_dir}")

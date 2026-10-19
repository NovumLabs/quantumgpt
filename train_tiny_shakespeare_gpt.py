"""Train a small character-level GPT model from scratch on Tiny Shakespeare.

For Google Colab, select a GPU runtime if one is available.

This standalone educational script includes the full small-model workflow:
data loading, character-level tokenization, model definition, training, and text
generation. It is an introductory experiment, not a prerequisite for the main
physics-model pipeline.
"""
import os
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------
# 1. Settings (hyperparameters). Adjust these values when experimenting.
# ------------------------------------------------------------
batch_size = 32       # Number of examples processed in one batch.
block_size = 128      # Number of previous characters the model can attend to (context length).
max_iters = 3000      # Number of training iterations.
eval_interval = 500   # Print the loss every this many steps.
learning_rate = 3e-4
n_embd = 128          # Embedding dimension for each character.
n_head = 4            # Number of attention heads.
n_layer = 4           # Number of Transformer blocks.
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print("Device:", device)

# ------------------------------------------------------------
# 2. Data: Tiny Shakespeare (a small text file, approximately 1 MB).
# ------------------------------------------------------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(URL, "input.txt")
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("Total characters:", len(text))

# ------------------------------------------------------------
# 3. Character-level tokenizer: each character maps to an integer.
# ------------------------------------------------------------
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # String -> integer ID.
itos = {i: ch for ch, i in stoi.items()}       # Integer ID -> string.


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))                       # 90% training data, 10% validation data.
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])  # y is x shifted forward by one character.
    return x.to(device), y.to(device)


# ------------------------------------------------------------
# 4. Model
# ------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Each character can attend to earlier characters, but not to future characters."""

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
    """A Transformer block consists of attention and a feed-forward network (MLP)."""

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


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)    # Character ID -> embedding vector.
        self.pos_emb = nn.Embedding(block_size, n_embd)    # Position ID -> positional embedding vector.
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)       # Hidden vector -> scores for the next character.

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = TinyGPT().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ------------------------------------------------------------
# 6. Generate sample text.
# ------------------------------------------------------------
start = torch.zeros((1, 1), dtype=torch.long, device=device)  # A single starting token.
out = model.generate(start, max_new_tokens=500)
print("\n----- GENERATED TEXT -----")
print(decode(out[0].tolist()))

torch.save(model.state_dict(), "tiny_gpt.pt")
print("\nModel saved: tiny_gpt.pt")

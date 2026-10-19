"""
Interactive QA test for the QuantumGPT GPT-2-style checkpoint.

Expected checkpoint keys:
  tok_emb.weight, pos_emb.weight, blocks.N.*, ln_f.*, lm_head.weight

Run:
  python test_quantumgpt_qa_fixed.py
  python test_quantumgpt_qa_fixed.py --checkpoint "E:/path/to/qm_gpt(3).pt"
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


# Architecture inferred from the supplied qm_gpt(3).pt checkpoint.
VOCAB_SIZE = 50304
BLOCK_SIZE = 256
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
FFN_DIM = 1024


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        output = output.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, dim
        )
        return self.proj(output)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention(D_MODEL, N_HEADS)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, D_MODEL),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class QuantumGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Keep only the most recent context window.
        input_ids = input_ids[:, -BLOCK_SIZE:]
        seq_len = input_ids.shape[1]

        positions = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.ln_f(x))


def generate_answer(
    model: QuantumGPT,
    tokenizer: GPT2TokenizerFast,
    question: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    prompt = f"Question: {question.strip()}\nAnswer:"
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt"
    ).to(device)
    generated_ids = input_ids.clone()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = generated_ids[:, -BLOCK_SIZE:]
            logits = model(context)[:, -1, :].float()

            # The GPT-2 tokenizer has 50,257 IDs; remaining rows are padding.
            logits[:, tokenizer.vocab_size:] = float("-inf")

            if temperature <= 0:
                next_token = torch.argmax(
                    logits, dim=-1, keepdim=True
                )
            else:
                logits = logits / temperature

                if top_k > 0:
                    k = min(top_k, tokenizer.vocab_size)
                    top_values = torch.topk(
                        logits, k, dim=-1
                    ).values
                    cutoff = top_values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(
                        logits < cutoff, float("-inf")
                    )

                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1
                )

            generated_ids = torch.cat(
                [generated_ids, next_token], dim=1
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

    answer_ids = generated_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(
        answer_ids, skip_special_tokens=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively test QuantumGPT question-answer generation."
    )
    parser.add_argument(
        "--checkpoint",
        default="qm_gpt(3).pt",
        help="Path to the trained checkpoint (default: qm_gpt(3).pt)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature; use 0 for greedy decoding (default: 0.8)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling limit; use 0 to disable top-k (default: 40)",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}\n"
            "Copy qm_gpt(3).pt beside this script, or pass its full path "
            "using --checkpoint."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")

    # This checkpoint is a model state_dict, not an optimizer-resume bundle.
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint must contain a model state dictionary.")
    if "tok_emb.weight" not in state_dict:
        raise RuntimeError(
            "Unexpected checkpoint format: missing 'tok_emb.weight'. "
            "This script targets the qm_gpt(3).pt architecture."
        )

    model = QuantumGPT()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Model loaded successfully. Parameters: {parameter_count:,}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.vocab_size > VOCAB_SIZE:
        raise RuntimeError(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) is larger "
            f"than model vocabulary ({VOCAB_SIZE})."
        )

    print("Tokenizer: GPT-2")
    print("\nEnter a question. Type 'exit' or 'quit' to stop.")
    print("Note: this is a base language model; it may not follow QA instructions reliably.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            answer = generate_answer(
                model=model,
                tokenizer=tokenizer,
                question=question,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"\nQuantumGPT: {answer or '[No text generated]'}\n")
        except RuntimeError as exc:
            print(f"\nGeneration error: {exc}\n")


if __name__ == "__main__":
    main()

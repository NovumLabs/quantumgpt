"""Train a custom BPE tokenizer with an 8,192-token vocabulary.

The earlier model used approximately 17.7 million parameters, including about
12.9 million in its 50,304-by-256 embedding table. An 8,192-token vocabulary
makes that table roughly six times smaller, leaving more of the parameter
budget available for Transformer layers.

Trade-off: a smaller vocabulary usually represents the same text with more
tokens and therefore longer sequences. This script compares tokenization
behavior and embedding-table sizes against the GPT-2 tokenizer.

Required inputs: physics_clean.jsonl and physics_qa.jsonl.

Run:
    python -m pip install tokenizers datasets tiktoken
    python train_physics_bpe_tokenizer_8k.py

Output: tokenizer_8k.json
"""
import json
import os
import shutil
from itertools import islice

VOCAB_SIZE = 8192
N_WEB_TRAIN = 20_000     # Number of FineWeb-Edu documents used to train the tokenizer.
N_WEB_EVAL = 300         # Reserved for comparison; not used for tokenizer training.
N_QA_TRAIN = 5_000
N_QA_EVAL = 500
MAX_DOC_CHARS = 20_000   # Truncate very long documents to improve processing speed.
PHYSICS_REPEAT = 2       # Repeat physics documents so physics-related terms influence merge statistics more strongly.
OUT = "tokenizer_8k.json"
EOT = "<|endoftext|>"

if __name__ == "__main__":
    from datasets import load_dataset
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    # ---------------- Data ----------------
    print("Loading general English text from FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    web = [ex["text"][:MAX_DOC_CHARS] for ex in islice(ds, N_WEB_TRAIN + N_WEB_EVAL)]
    web_train, web_eval = web[:N_WEB_TRAIN], web[N_WEB_TRAIN:]

    with open("physics_clean.jsonl", "r", encoding="utf-8") as f:
        phys = [json.loads(line)["text"][:MAX_DOC_CHARS] for line in f]
    phys_eval = [t for i, t in enumerate(phys) if i % 20 == 0]        # Use the same validation-selection rule as Step 5a.
    phys_train = [t for i, t in enumerate(phys) if i % 20 != 0]

    with open("physics_qa.jsonl", "r", encoding="utf-8") as f:
        qa = [f"Question: {d['question'].strip()}\nAnswer: {d['answer'].strip()}"
              for d in (json.loads(line) for line in f)]
    qa_train, qa_eval = qa[:N_QA_TRAIN], qa[N_QA_TRAIN : N_QA_TRAIN + N_QA_EVAL]

    print(f"web {len(web_train):,} | physics docs {len(phys_train):,} (x{PHYSICS_REPEAT}) | Q&A {len(qa_train):,}")

    def train_iter():
        yield from web_train
        for _ in range(PHYSICS_REPEAT):
            yield from phys_train
        yield from qa_train

    total = len(web_train) + PHYSICS_REPEAT * len(phys_train) + len(qa_train)

    # ---------------- Train ----------------
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # All 256 byte values are included, allowing the tokenizer to represent arbitrary text.
        show_progress=True,
    )
    tok.train_from_iterator(train_iter(), trainer=trainer, length=total)
    tok.save(OUT)
    print(f"\nSaved {OUT}. Vocab size: {tok.get_vocab_size()}, {EOT} id: {tok.token_to_id(EOT)}")

    # ---------------- Sanity check: encode -> decode ----------------
    s = "The Schrödinger equation: iħ ∂ψ/∂t = Ĥψ describes how a quantum state evolves."
    ids = tok.encode(s).ids
    assert tok.decode(ids) == s, "Encode/decode round-trip failed."
    print("Encode/decode round-trip passed. Sentence:", len(s.encode('utf-8')), "bytes ->", len(ids), "tokens")
    print("Tokens:", tok.encode(s).tokens)   # The 'Ġ' marker represents a space before a word.

    # ---------------- Compare with GPT-2's tokenizer ----------------
    def bytes_per_token(texts, encode_fn):
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        ntok = sum(len(encode_fn(t)) for t in texts)
        return nbytes / ntok

    ours = lambda t: tok.encode(t).ids
    try:
        import tiktoken

        gpt2 = tiktoken.get_encoding("gpt2")
        gpt2_fn = gpt2.encode_ordinary
    except ImportError:
        gpt2_fn = None

    print("\nBytes per token (higher values indicate more compact tokenization and shorter sequences):")
    print(f"{'held-out set':<22}{'ours (8k)':>12}{'GPT-2 (50k)':>14}")
    for name, texts in [("general web", web_eval), ("physics docs", phys_eval), ("physics Q&A", qa_eval)]:
        b_ours = bytes_per_token(texts, ours)
        b_gpt2 = f"{bytes_per_token(texts, gpt2_fn):.2f}" if gpt2_fn else "n/a"
        print(f"{name:<22}{b_ours:>12.2f}{b_gpt2:>14}")

    print("\nEmbedding-table parameters (vocabulary size × embedding dimension):")
    for d in (256, 512, 768):
        print(f"  n_embd={d}:  ours {VOCAB_SIZE * d / 1e6:5.1f}M   vs   50k vocab {50304 * d / 1e6:5.1f}M")

    # ---------------- Drive backup ----------------
    # Optional backup. Set QM_DRIVE_DIR explicitly after mounting Drive in Colab.
    drive_dir = os.environ.get("QM_DRIVE_DIR", "")
    if drive_dir and os.path.isdir(os.path.dirname(drive_dir)):
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy2(OUT, os.path.join(drive_dir, OUT))
        print(f"\nBacked up tokenizer to: {drive_dir}")

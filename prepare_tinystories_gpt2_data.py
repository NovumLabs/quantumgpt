"""Prepare TinyStories data using the GPT-2 tokenizer.

TinyStories contains short, simple English stories. It is useful for initial
language-model experiments because a small model can learn basic language
patterns from a relatively compact dataset.

Run in Colab or another Python environment:
    python -m pip install datasets tiktoken
    python prepare_tinystories_gpt2_data.py

Outputs:
    train.bin and val.bin, containing token IDs stored as uint16 values.
"""
from itertools import islice

import numpy as np
import tiktoken
from datasets import load_dataset

N_TRAIN = 100_000   # Number of stories to use (approximately 20M tokens); increase this for a larger dataset.
N_VAL = 2_000

# Step 2 introduced a custom BPE implementation. This stage uses the faster Rust-backed GPT-2 tokenizer (tiktoken),
# because the pure-Python BPE implementation would be too slow for a dataset of this size.
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # <|endoftext|>: separator between stories.


def tokenize_split(split, n, out_path):
    # streaming=True reads stories incrementally instead of downloading the entire dataset (approximately 2 GB).
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    chunks = []
    for i, ex in enumerate(islice(ds, n)):
        toks = enc.encode_ordinary(ex["text"])
        toks.append(EOT)
        chunks.append(np.array(toks, dtype=np.uint16))  # 50,257 is less than 65,536, so uint16 can store these token IDs.
        if (i + 1) % 20_000 == 0:
            print(f"  {split}: {i + 1} stories done")
    arr = np.concatenate(chunks)
    arr.tofile(out_path)
    print(f"{out_path}: {len(arr):,} tokens")


tokenize_split("train", N_TRAIN, "train.bin")
tokenize_split("validation", N_VAL, "val.bin")

# Preview one story.
ids = np.fromfile("train.bin", dtype=np.uint16)[:150].tolist()
print("\nDecoded preview of the first 150 tokens:\n")
print(enc.decode(ids))

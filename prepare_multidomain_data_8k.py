"""Tokenize TinyStories, physics documents, and Q&A with the 8k tokenizer.

Token IDs are specific to a tokenizer. Existing .bin files produced by the
GPT-2 tokenizer cannot be reused with tokenizer_8k.json because the same ID may
represent a different token. This stage therefore creates separately named
8k-tokenizer files and does not overwrite the GPT-2-tokenizer files.

Requirements: tokenizer_8k.json, physics_clean.jsonl, and internet access to
stream TinyStories and the CAMEL physics records.

Run:
    python -m pip install tokenizers datasets
    python prepare_multidomain_data_8k.py

Outputs:
    stories_train_8k.bin  stories_val_8k.bin
    web_train_8k.bin      web_val_8k.bin
    qa_train_8k.bin       qa_val_8k.bin
    qm_train_8k.bin       qm_val_8k.bin
"""
import hashlib
import json
from itertools import islice

import numpy as np

TOKENIZER_FILE = "tokenizer_8k.json"
N_STORIES_TRAIN = 100_000
N_STORIES_VAL = 2_000
QM_TOPIC_MATCH = "quantum mechanics"
HELDOUT_PCT = 5
HELDOUT_PCT_QM = 20


def format_qa(question, answer):
    return f"Question: {question.strip()}\nAnswer: {answer.strip()}"


def is_heldout(topic, sub_topic, pct):
    h = int(hashlib.md5(f"{topic}|{sub_topic}".encode("utf-8")).hexdigest(), 16) % 100
    return h < pct


if __name__ == "__main__":
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_FILE)
    EOT = tok.token_to_id("<|endoftext|>")
    if EOT is None:
        raise ValueError("tokenizer_8k.json does not contain the required <|endoftext|> token")
    if tok.get_vocab_size() > 65536:
        raise ValueError("Vocabulary is too large for uint16 token files")
    print(f"Tokenizer loaded. Vocab {tok.get_vocab_size()}, EOT id {EOT}")

    def encode_texts(texts):
        chunks = []
        for t in texts:
            ids = tok.encode(t).ids
            ids.append(EOT)
            chunks.append(np.array(ids, dtype=np.uint16))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)

    def save(name, texts):
        arr = encode_texts(texts)
        arr.tofile(name)
        print(f"{name:<16} {len(texts):>6,} texts  {len(arr):>12,} tokens")

    # ---------------- TinyStories ----------------
    print("\nLoading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stories_train = [ex["text"] for ex in islice(ds, N_STORIES_TRAIN)]
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    stories_val = [ex["text"] for ex in islice(ds_val, N_STORIES_VAL)]
    save("stories_train_8k.bin", stories_train)
    save("stories_val_8k.bin", stories_val)

    # ---------------- Physics web docs (Step 4b-r output) ----------------
    with open("physics_clean.jsonl", "r", encoding="utf-8") as f:
        web = [json.loads(line)["text"] for line in f]
    web_val = [t for i, t in enumerate(web) if i % 20 == 0]
    web_train = [t for i, t in enumerate(web) if i % 20 != 0]
    save("web_train_8k.bin", web_train)
    save("web_val_8k.bin", web_val)

    # ---------------- CAMEL physics Q&A ----------------
    # Load the original CAMEL records again because physics_qa.jsonl contains only
    # question/answer fields; the source topic/sub_topic metadata is required for
    # a group-aware train/validation split. This stage therefore uses the source
    # dataset's rows for splitting, not the simplified JSONL rows.
    rows = list(load_dataset("dim/camel_ai_physics", split="train"))

    def topic_of(r):
        return (r.get("topic;") or r.get("topic") or "").strip()

    qa_train, qa_val, qm_train, qm_val = [], [], [], []
    for r in rows:
        text = format_qa(r["message_1"], r["message_2"])
        is_qm = QM_TOPIC_MATCH in topic_of(r).lower()
        pct = HELDOUT_PCT_QM if is_qm else HELDOUT_PCT
        held = is_heldout(topic_of(r), r.get("sub_topic", ""), pct)
        (qa_val if held else qa_train).append(text)
        if is_qm:
            (qm_val if held else qm_train).append(text)

    save("qa_train_8k.bin", qa_train)
    save("qa_val_8k.bin", qa_val)
    save("qm_train_8k.bin", qm_train)
    save("qm_val_8k.bin", qm_val)

    import os
    import shutil

    # Optional backup. Set QM_DRIVE_DIR explicitly after mounting Drive in Colab.
    drive_dir = os.environ.get("QM_DRIVE_DIR", "")
    if drive_dir and os.path.isdir(os.path.dirname(drive_dir)):
        os.makedirs(drive_dir, exist_ok=True)
        for f in ["stories_train_8k.bin", "stories_val_8k.bin", "web_train_8k.bin", "web_val_8k.bin",
                  "qa_train_8k.bin", "qa_val_8k.bin", "qm_train_8k.bin", "qm_val_8k.bin"]:
            shutil.copy2(f, os.path.join(drive_dir, f))
        print(f"\nBacked up 8k data files to: {drive_dir}")

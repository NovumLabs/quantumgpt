"""Prepare physics documents and Q&A for GPT-2-tokenizer training.

Two physics sources are used:
  1. Web documents from physics_clean.jsonl.
  2. CAMEL physics question-answer pairs, which provide more text than the web
     documents alone.

Q&A records are serialized in this format so the model can learn the
question-answer pattern:
    Question: <problem>
    Answer: <solution><|endoftext|>

Validation split policy:
  CAMEL contains many related problems grouped by topic and sub-topic. A random
  record-level split could place near-duplicates in both training and validation
  sets, making validation loss look artificially good. This script therefore
  keeps each topic/sub-topic group on one side of the split.

Run:
    python -m pip install datasets tiktoken
    python prepare_physics_training_data.py

Outputs:
    web_train.bin  web_val.bin
    qa_train.bin   qa_val.bin
    qm_train.bin   qm_val.bin  — quantum-mechanics Q&A for later experiments
"""
import hashlib
import json
import re

import numpy as np

WEB_FILE = "physics_clean.jsonl"
QA_DATASET = "dim/camel_ai_physics"
QM_TOPIC_MATCH = "quantum mechanics"   # Treat a record as QM-related when its topic contains this phrase.
HELDOUT_PCT = 5        # Hold out approximately 5% of topic groups for validation.
HELDOUT_PCT_QM = 20    # QM has fewer groups (approximately 25), so reserve a larger share for validation.

QM_FALLBACK = re.compile(r"schr(?:o|ö)dinger|wave ?function|hamiltonian|uncertainty principle|quantum", re.I)


def format_qa(question, answer):
    return f"Question: {question.strip()}\nAnswer: {answer.strip()}"


def is_heldout(topic, sub_topic, pct):
    """Hash each (topic, sub_topic) group so it always goes to the same split, making the split stable."""
    h = int(hashlib.md5(f"{topic}|{sub_topic}".encode("utf-8")).hexdigest(), 16) % 100
    return h < pct


if __name__ == "__main__":
    from collections import Counter

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token

    def encode_texts(texts):
        chunks = []
        for t in texts:
            ids = enc.encode_ordinary(t)
            ids.append(EOT)
            chunks.append(np.array(ids, dtype=np.uint16))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)

    def save(name, texts):
        arr = encode_texts(texts)
        arr.tofile(name)
        print(f"{name:<14} {len(texts):>6,} texts  {len(arr):>12,} tokens")

    # ---------------- Web docs ----------------
    with open(WEB_FILE, "r", encoding="utf-8") as f:
        web = [json.loads(line)["text"] for line in f]
    web_val = [t for i, t in enumerate(web) if i % 20 == 0]      # Approximately 5% for validation.
    web_train = [t for i, t in enumerate(web) if i % 20 != 0]

    # ---------------- CAMEL Q&A ----------------
    rows = list(load_dataset(QA_DATASET, split="train"))

    def topic_of(r):
        return (r.get("topic;") or r.get("topic") or "").strip()   # In this dataset mirror, the column is named "topic;".

    topics = Counter(topic_of(r) for r in rows)
    print("Physics topics (with pair counts):")
    for t, c in topics.most_common():
        print(f"  {c:>5}  {t}")

    qm_flags = [QM_TOPIC_MATCH in topic_of(r).lower() for r in rows]
    if not any(qm_flags):
        print(f"\nWARNING: no topic matched '{QM_TOPIC_MATCH}' pai ni. classifying QM examples using question keywords instead.")
        qm_flags = [bool(QM_FALLBACK.search(r["message_1"])) for r in rows]

    qa_train, qa_val, qm_train, qm_val = [], [], [], []
    for r, is_qm in zip(rows, qm_flags):
        text = format_qa(r["message_1"], r["message_2"])
        pct = HELDOUT_PCT_QM if is_qm else HELDOUT_PCT
        held = is_heldout(topic_of(r), r.get("sub_topic", ""), pct)
        (qa_val if held else qa_train).append(text)
        if is_qm:
            (qm_val if held else qm_train).append(text)

    print()
    save("web_train.bin", web_train)
    save("web_val.bin", web_val)
    save("qa_train.bin", qa_train)
    save("qa_val.bin", qa_val)
    save("qm_train.bin", qm_train)
    save("qm_val.bin", qm_val)
    if len(qm_val) == 0:
        print("WARNING: qm_val is empty. Increase HELDOUT_PCT_QM.")

    print("\nExample of a formatted Q&A record:\n")
    print(qa_train[0][:600])

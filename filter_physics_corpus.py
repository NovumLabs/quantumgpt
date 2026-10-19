"""Apply a quality filter to the collected physics corpus.

This stage runs locally and does not require internet access.

The initial collection stage exposed three issues:
  1. Web boilerplate, such as navigation links and publication notices.
  2. Documents that are not prose, such as syllabi, navigation lists, and glossaries.
  3. An overly broad quantum-mechanics score; terms such as "spin" and "orbital"
     can also appear in astronomy content.

This script reads physics_docs.jsonl and writes:
  physics_clean.jsonl — filtered physics documents
  qm_docs.jsonl       — documents with stronger quantum-mechanics signals

It also reports why documents were rejected so the thresholds can be reviewed.

Run:
    python filter_physics_corpus.py
"""
import json
import random
import re
from collections import Counter, defaultdict

IN_FILE = "physics_docs.jsonl"
OUT_CLEAN = "physics_clean.jsonl"
OUT_QM = "qm_docs.jsonl"

MIN_PHYS_SCORE = 10   # Step 4 used 8. A higher threshold retains fewer documents but may improve physics density.
MIN_CHARS = 1_000
MIN_WORDS = 150
QM_MIN_CORE_TERMS = 4   # Minimum number of distinct QM-specific terms.
QM_MIN_QUANTUM_COUNT = 3  # Minimum number of occurrences of the word "quantum".

# ------------------------------------------------------------
# 1. Remove line-level boilerplate.
# ------------------------------------------------------------
BOILERPLATE = re.compile(
    r"^(skip to|posted on|copyright|©|all rights reserved|click here|share (this|on)|"
    r"sign up|subscribe|log ?in|follow us|read more|related (articles|posts)|tags?:|categories?:)"
    r"|\b(all rights reserved|privacy policy|terms of (use|service))\b",
    re.I,
)


def clean_lines(text):
    keep = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            keep.append("")
            continue
        if s.count("|") >= 3:                          # Navigation box or table.
            continue
        if len(s) < 200 and BOILERPLATE.search(s):     # Short boilerplate line.
            continue
        keep.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


# ------------------------------------------------------------
# 2. Document-level quality checks (a simplified version of Gopher/C4-style filters).
# ------------------------------------------------------------
STOP = {
    "the", "of", "and", "to", "a", "in", "is", "that", "it", "for", "as", "with", "was",
    "on", "are", "be", "by", "this", "an", "which", "or", "from", "at", "have", "not",
    "can", "has", "we", "you", "their", "its", "these", "when",
}


def quality_reject(text):
    """Return a rejection reason for a low-quality document, or None if it passes."""
    if len(text) < MIN_CHARS:
        return "too_short"
    words = text.split()
    n = len(words)
    if n < MIN_WORDS:
        return "too_few_words"
    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 8:
        return "odd_word_length"
    if sum(any(c.isalpha() for c in w) for w in words) / n < 0.8:
        return "too_many_symbols"
    stop_ratio = sum(w.lower().strip(".,;:!?()\"'") in STOP for w in words) / n
    if stop_ratio < 0.2:
        return "not_prose"       # Word lists, glossaries, and lists of names.
    if text.count("|") / len(text) * 1000 > 3:
        return "tables"
    return None


# ------------------------------------------------------------
# 3. Apply a stricter quantum-mechanics relevance flag.
#    Use word-boundary matching and exclude ambiguous terms such as "spin", "orbital", and "electron".
# ------------------------------------------------------------
QM_CORE = [
    r"quantum", r"wave ?functions?", r"schr(?:o|ö)dinger", r"uncertainty principle",
    r"heisenberg", r"eigenvalues?", r"eigenstates?", r"hamiltonian", r"superposition",
    r"entanglement", r"harmonic oscillator", r"tunnell?ing", r"commutators?",
    r"planck'?s constant", r"dirac", r"probability amplitude", r"quantum numbers?",
    r"wave-particle duality",
]
QM_CORE_RE = [re.compile(r"\b" + p + r"\b", re.I) for p in QM_CORE]
QUANTUM_RE = re.compile(r"\bquantum\b", re.I)


def qm_core_score(text):
    return sum(1 for r in QM_CORE_RE if r.search(text))


def is_qm_doc(text):
    return (qm_core_score(text) >= QM_MIN_CORE_TERMS
            and len(QUANTUM_RE.findall(text)) >= QM_MIN_QUANTUM_COUNT)


def preview(text, n=220):
    return text[:n].replace("\n", " ") + ("..." if len(text) > n else "")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    with open(IN_FILE, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    print(f"Input docs: {len(docs):,}")

    print("\nNumber of documents retained at each phys_score threshold:")
    for t in (8, 10, 12, 15, 20):
        print(f"  score >= {t:>2}: {sum(d['phys_score'] >= t for d in docs):,}")

    reasons = Counter()
    examples = defaultdict(list)
    kept = []
    for d in docs:
        if d["phys_score"] < MIN_PHYS_SCORE:
            reasons["low_phys_score"] += 1
            continue
        text = clean_lines(d["text"])
        why = quality_reject(text)
        if why:
            reasons[why] += 1
            if len(examples[why]) < 2:
                examples[why].append(preview(text or d["text"]))
            continue
        kept.append({"text": text, "phys_score": d["phys_score"],
                     "qm_core": qm_core_score(text), "is_qm": is_qm_doc(text)})

    with open(OUT_CLEAN, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    qm = [d for d in kept if d["is_qm"]]
    with open(OUT_QM, "w", encoding="utf-8") as f:
        for d in qm:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    chars = sum(len(d["text"]) for d in kept)
    qm_chars = sum(len(d["text"]) for d in qm)
    print("\n===== RESULTS =====")
    print(f"Kept docs     : {len(kept):,}  (~{chars // 4:,} tokens)")
    print(f"QM docs       : {len(qm):,}  (~{qm_chars // 4:,} tokens)")
    print("\nRejection reasons (document counts):")
    for why, c in reasons.most_common():
        print(f"  {why:<18} {c:,}")

    print("\n----- Examples of rejected documents (inspect whether each rejection is appropriate) -----")
    for why, exs in examples.items():
        for e in exs:
            print(f"[{why}] {e}\n")

    random.seed(1)
    print("----- Three QM documents -----")
    for d in random.sample(qm, min(3, len(qm))):
        print(f"[phys {d['phys_score']}, qm_core {d['qm_core']}] {preview(d['text'])}\n")

"""Collect candidate physics documents and physics question-answer data.

This stage does not train a model. It builds candidate datasets because data
quality is a major factor in language-model quality.

What it does:
  1. Streams FineWeb-Edu and filters documents using physics-related terms.
  2. Normalizes whitespace and removes exact duplicates and out-of-range documents.
  3. Computes physics and quantum-mechanics relevance scores for each document.
  4. Saves synthetic CAMEL physics question-answer pairs separately for later use.

Run:
    python -m pip install datasets
    python collect_physics_corpus.py

Outputs:
    physics_docs.jsonl  — candidate documents with relevance scores
    physics_qa.jsonl    — question-answer records
"""
import hashlib
import json
import random
import re

# ------------------------------------------------------------
# Settings: lower MIN_PHYS_SCORE if the yield is too low; raise it if too much noise remains.
# ------------------------------------------------------------
MAX_DOCS_SCAN = 500_000   # Maximum number of web documents to scan (up to about 2 GB of streamed text).
TARGET_KEPT = 10_000      # Stop after this many physics documents have been retained.
MIN_PHYS_SCORE = 8        # Minimum number of distinct physics terms required.
MIN_CHARS = 1_000
MAX_CHARS = 30_000
QM_MIN_SCORE = 4          # Minimum number of distinct QM terms required to classify a document as QM-related.

PHYSICS_TERMS = [
    "physics", "physicist", "quantum", "electron", "proton", "neutron", "photon",
    "momentum", "kinetic energy", "potential energy", "thermodynamic", "entropy",
    "relativity", "magnetic field", "electric field", "electromagnetic", "wavelength",
    "wave function", "newton", "joule", "velocity", "acceleration", "oscillat",
    "nucleus", "atomic", "gravitational", "friction", "voltage", "equilibrium",
    "conservation of energy", "angular momentum", "frequency", "amplitude", "torque",
    "particle", "radiation", "spectrum", "einstein", "planck", "boltzmann",
]

QM_TERMS = [
    "quantum", "wave function", "wavefunction", "schrodinger", "schrödinger",
    "superposition", "uncertainty principle", "heisenberg", "eigenvalue", "eigenstate",
    "hamiltonian", "planck", "photon", "electron", "spin", "entanglement", "observable",
    "probability amplitude", "orbital", "harmonic oscillator", "tunneling", "tunnelling",
    "bohr", "dirac", "commutator", "wave-particle", "quantum number",
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def score(text_lower, terms):
    """Count how many distinct terms appear in the text."""
    return sum(1 for t in terms if t in text_lower)


def clean(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)        # Collapse repeated spaces and tabs into one space.
    text = re.sub(r" *\n *", "\n", text)       # Remove whitespace from both ends of each line.
    text = re.sub(r"\n{3,}", "\n\n", text)     # Replace three or more consecutive blank lines with two.
    return text.strip()


def doc_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def preview(text, n=300):
    return text[:n].replace("\n", " ") + ("..." if len(text) > n else "")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    from datasets import load_dataset

    # ---------- Part A: Collect physics documents from FineWeb-Edu ----------
    print("Streaming FineWeb-Edu; this may take several minutes...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    seen = set()
    kept = []
    scanned = 0
    for ex in ds:
        scanned += 1
        text = ex["text"]
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            low = text.lower()
            p = score(low, PHYSICS_TERMS)
            if p >= MIN_PHYS_SCORE:
                text = clean(text)
                h = doc_hash(text)
                if h not in seen:
                    seen.add(h)
                    kept.append({"text": text, "phys_score": p, "qm_score": score(low, QM_TERMS)})
        if scanned % 25_000 == 0:
            print(f"  scanned {scanned:,} docs, kept {len(kept):,}")
        if scanned >= MAX_DOCS_SCAN or len(kept) >= TARGET_KEPT:
            break

    with open("physics_docs.jsonl", "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    total_chars = sum(len(d["text"]) for d in kept)
    qm_docs = [d for d in kept if d["qm_score"] >= QM_MIN_SCORE]
    print("\n===== RESULTS =====")
    print(f"Scanned docs : {scanned:,}")
    print(f"Kept docs    : {len(kept):,}  ({100 * len(kept) / max(1, scanned):.2f}% of scanned)")
    print(f"Total chars  : {total_chars:,}  (~{total_chars // 4:,} tokens)")
    print(f"QM-heavy docs: {len(qm_docs):,}  (qm_score >= {QM_MIN_SCORE})")

    # Inspect representative examples; this is one of the most useful quality checks.
    random.seed(0)
    print("\n----- Three random physics documents -----")
    for d in random.sample(kept, min(3, len(kept))):
        print(f"[phys {d['phys_score']}, qm {d['qm_score']}] {preview(d['text'])}\n")
    print("----- Three documents with the highest QM scores -----")
    for d in sorted(kept, key=lambda d: d["qm_score"], reverse=True)[:3]:
        print(f"[phys {d['phys_score']}, qm {d['qm_score']}] {preview(d['text'])}\n")

    # ---------- Part B: Collect CAMEL physics Q&A ----------
    # Synthetic physics problem-solution pairs; these may be useful for later instruction tuning.
    try:
        qa = load_dataset("dim/camel_ai_physics", split="train")
        with open("physics_qa.jsonl", "w", encoding="utf-8") as f:
            for row in qa:
                f.write(json.dumps({"question": row["message_1"], "answer": row["message_2"]},
                                   ensure_ascii=False) + "\n")
        print(f"physics_qa.jsonl: {len(qa):,} Q&A pairs saved")
        print("Example question:", preview(qa[0]["message_1"], 200))
    except Exception as e:
        print("Could not load the Q&A dataset; you can try again later:", e)

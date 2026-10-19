"""Train a small Byte Pair Encoding (BPE) tokenizer from scratch.

BPE repeatedly merges frequent adjacent token pairs into new tokens. Common
sequences such as "the", "ing", or "quantum" may then be represented by a
single token. This educational implementation is written in pure Python and
does not require PyTorch.
"""
import os
import urllib.request

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def get_stats(ids):
    """Count how often each adjacent token pair occurs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace each occurrence of pair (a, b) in ids with new_id."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ------------------------------------------------------------
# Tokenizer class
# ------------------------------------------------------------


class BPETokenizer:
    def train(self, text, vocab_size, verbose=True):
        assert vocab_size >= 256
        # Start with one token for each byte value (0–255).
        ids = list(text.encode("utf-8"))
        self.merges = {}  # (a, b) -> new token ID.
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # Most frequent adjacent pair.
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            if verbose and (i < 10 or (i + 1) % 50 == 0):
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} (count {stats[pair]})")
        # Token ID -> bytes lookup table used for decoding.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Apply the pair that was merged earliest during tokenizer training.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # No learned merge applies to the remaining sequence.
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("input.txt"):
        urllib.request.urlretrieve(URL, "input.txt")
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # The pure-Python implementation is slow, so train on only the first 200,000 characters.
    train_text = text[:200_000]

    tok = BPETokenizer()
    tok.train(train_text, vocab_size=512)  # 256 byte + 256 merge

    # Test 1: verify that encoding and decoding reproduce the original text.
    sample = "First Citizen: Before we proceed any further, hear me speak."
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    print("\nOriginal length (bytes):", len(sample.encode("utf-8")))
    print("Token count after BPE  :", len(ids))
    print("Tokens:", [tok.vocab[i].decode("utf-8", errors="replace") for i in ids])

    # Test 2: inspect performance on physics text; the tokenizer was trained on Shakespeare, so results may be poor.
    physics = "The Schrodinger equation describes how the quantum state of a system evolves."
    pids = tok.encode(physics)
    print("\nPhysics sentence bytes:", len(physics.encode("utf-8")), "-> tokens:", len(pids))
    print("If trained on physics text, terms such as 'quantum' and 'equation' may become individual tokens.")

    # Optional: compare with the original GPT-2 tokenizer (requires `pip install tiktoken`).
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        print("\nGPT-2 tokenizer (50,257 vocab):")
        print("  Shakespeare sentence:", len(enc.encode(sample)), "tokens")
        print("  Physics sentence    :", len(enc.encode(physics)), "tokens")
    except ImportError:
        print("\n(Optional: install `tiktoken` to compare results with the GPT-2 tokenizer.)")

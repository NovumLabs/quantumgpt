"""Small offline tests for the educational BPE tokenizer."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("step2_bpe", ROOT / "train_bpe_tokenizer_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BPETokenizerTests(unittest.TestCase):
    def test_encode_decode_round_trip_ascii_and_unicode(self):
        tokenizer = MODULE.BPETokenizer()
        tokenizer.train(("physics quantum wave function Schrödinger " * 30), vocab_size=320, verbose=False)
        samples = [
            "Quantum mechanics describes matter and energy.",
            "Schrödinger equation: ψ and ħ.",
            "",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)


if __name__ == "__main__":
    unittest.main()

"""Check the QuantumGPT development environment without modifying it.

Run from the repository root:
    python check_environment.py

This script is diagnostic only. It does not install packages, download datasets,
start training, or modify checkpoints.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = {
    "torch": "PyTorch (training and model execution)",
    "numpy": "NumPy (token arrays and memory-mapped datasets)",
    "datasets": "Hugging Face Datasets (dataset streaming/loading)",
    "tiktoken": "tiktoken (GPT-2 tokenizer for the 50k-vocabulary pipeline)",
    "tokenizers": "Hugging Face Tokenizers (the experimental 8k tokenizer)",
}
OPTIONAL_MODULES = {"psutil": "RAM reporting"}
EXPECTED_FILES = [
    "prepare_tinystories_gpt2_data.py",
    "train_tinystories_gpt2.py",
    "collect_physics_corpus.py",
    "filter_physics_corpus.py",
    "prepare_physics_training_data.py",
    "train_physics_gpt2.py",
    "specialize_quantum_gpt2.py",
    "train_physics_bpe_tokenizer_8k.py",
    "prepare_multidomain_data_8k.py",
    "train_english_gpt_8k.py",
    "requirements.txt",
]


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    print("=" * 72)
    print("QUANTUMGPT — ENVIRONMENT CHECK")
    print("=" * 72)
    print(f"Project root : {ROOT}")
    print(f"Operating OS : {platform.system()} {platform.release()}")
    print(f"Python       : {platform.python_version()} ({sys.executable})")
    print(f"Working dir  : {Path.cwd()}")

    try:
        usage = shutil.disk_usage(ROOT)
        print(f"Free disk    : {usage.free / (1024**3):.2f} GiB")
        if usage.free < 10 * 1024**3:
            print("WARNING      : Less than 10 GiB free. Dataset and checkpoint operations may fail.")
    except OSError as exc:
        print(f"Free disk    : unavailable ({exc})")

    if module_available("psutil"):
        import psutil
        print(f"System RAM   : {psutil.virtual_memory().total / (1024**3):.2f} GiB")
    else:
        print("System RAM   : unavailable (optional: install psutil)")

    print("\nDependencies")
    missing = []
    for name, purpose in REQUIRED_MODULES.items():
        available = module_available(name)
        print(f"  {'OK' if available else 'MISSING':7} {name:12} — {purpose}")
        if not available:
            missing.append(name)
    for name, purpose in OPTIONAL_MODULES.items():
        print(f"  {'OK' if module_available(name) else 'OPTIONAL':7} {name:12} — {purpose}")

    if module_available("torch"):
        try:
            import torch
            print(f"\nPyTorch      : {torch.__version__}")
            print(f"CUDA         : {'available' if torch.cuda.is_available() else 'not available'}")
            if torch.cuda.is_available():
                print(f"GPU          : {torch.cuda.get_device_name(0)}")
                print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
            mps_backend = getattr(torch.backends, "mps", None)
            print(f"Apple MPS    : {'available' if mps_backend and mps_backend.is_available() else 'not available'}")
        except Exception as exc:  # diagnostic should report, not crash on an unusual torch install
            print(f"PyTorch check: failed ({type(exc).__name__}: {exc})")

    print("\nExpected project files")
    missing_files = []
    for name in EXPECTED_FILES:
        exists = (ROOT / name).is_file()
        print(f"  {'OK' if exists else 'MISSING':7} {name}")
        if not exists:
            missing_files.append(name)

    print("\nNotes")
    print("  • This check does not verify that datasets or checkpoints are compatible.")
    print("  • The GPT-2-tokenizer and 8k-tokenizer experiments use different token IDs.")
    print("  • Do not mix their binary token files or load checkpoints across architectures.")
    print("  • CPU execution is possible, but full training runs may be very slow.")

    if missing:
        print("\nRESULT: Setup incomplete. Install dependencies with: python -m pip install -r requirements.txt")
        return 1
    if missing_files:
        print("\nRESULT: Some expected scripts/files are missing. Check the repository root.")
        return 1
    print("\nRESULT: Required Python packages and expected project scripts are present.")
    print("This is not a guarantee that training can run; check each stage's input files first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

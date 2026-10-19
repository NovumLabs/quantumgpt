"""Safe, explicit stage runner for the current QuantumGPT flat-script workflow.

Examples:
    python pipeline_runner.py --list
    python pipeline_runner.py --stage prepare-stories-gpt2 --dry-run
    python pipeline_runner.py --stage prepare-stories-gpt2
    python pipeline_runner.py --stage train-stories-gpt2

Important:
- Exactly one stage is run per invocation; there is no automatic full-pipeline run.
- Existing outputs are never silently skipped or overwritten. Use --force only
  after you have reviewed/ backed up the listed files.
- Scripts currently read/write artifacts relative to the repository root. Keep
  the scripts in this root until their path handling is refactored together.
- train_tiny_shakespeare_gpt.py and train_bpe_tokenizer_demo.py are standalone educational experiments,
  not prerequisites for the current GPT-2-tokenizer training pipeline.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    script: str
    requires: tuple[str, ...]
    outputs: tuple[str, ...]
    description: str


STAGES: dict[str, Stage] = {
    "prepare-stories-gpt2": Stage(
        "prepare_tinystories_gpt2_data.py", (), ("train.bin", "val.bin"),
        "Tokenize TinyStories using the GPT-2 tokenizer.",
    ),
    "train-stories-gpt2": Stage(
        "train_tinystories_gpt2.py", ("train.bin", "val.bin"), ("tinystories_gpt.pt",),
        "Train the 50k-vocabulary baseline GPT model.",
    ),
    "collect-physics": Stage(
        "collect_physics_corpus.py", (), ("physics_docs.jsonl", "physics_qa.jsonl"),
        "Collect candidate physics documents and physics Q&A data.",
    ),
    "filter-physics": Stage(
        "filter_physics_corpus.py", ("physics_docs.jsonl",), ("physics_clean.jsonl", "qm_docs.jsonl"),
        "Filter collected documents and create physics/QM document files.",
    ),
    "prepare-physics-gpt2": Stage(
        "prepare_physics_training_data.py", ("physics_clean.jsonl",),
        ("web_train.bin", "web_val.bin", "qa_train.bin", "qa_val.bin", "qm_train.bin", "qm_val.bin"),
        "Prepare physics/Q&A token files for the GPT-2-tokenizer pipeline.",
    ),
    "train-physics-gpt2": Stage(
        "train_physics_gpt2.py",
        ("tinystories_gpt.pt", "train.bin", "val.bin", "web_train.bin", "web_val.bin",
         "qa_train.bin", "qa_val.bin", "qm_val.bin"),
        ("physics_gpt.pt",),
        "Continue training the GPT-2-tokenizer baseline on a mixed physics corpus.",
    ),
    "specialize-qm-gpt2": Stage(
        "specialize_quantum_gpt2.py",
        ("physics_gpt.pt", "train.bin", "val.bin", "qm_train.bin", "qm_val.bin",
         "qa_train.bin", "qa_val.bin", "web_val.bin"),
        ("qm_gpt.pt",),
        "Specialize the GPT-2-tokenizer physics model on quantum-mechanics data.",
    ),
    "train-tokenizer-8k": Stage(
        "train_physics_bpe_tokenizer_8k.py", ("physics_clean.jsonl", "physics_qa.jsonl"),
        ("tokenizer_8k.json",),
        "Train the separate experimental 8k-vocabulary tokenizer.",
    ),
    "prepare-data-8k": Stage(
        "prepare_multidomain_data_8k.py", ("tokenizer_8k.json", "physics_clean.jsonl"),
        ("stories_train_8k.bin", "stories_val_8k.bin", "web_train_8k.bin", "web_val_8k.bin",
         "qa_train_8k.bin", "qa_val_8k.bin", "qm_train_8k.bin", "qm_val_8k.bin"),
        "Prepare isolated 8k-tokenizer data files (does not overwrite GPT-2-tokenizer .bin files).",
    ),
    "train-english-8k": Stage(
        "train_english_gpt_8k.py", ("tokenizer_8k.json", "stories_train_8k.bin", "stories_val_8k.bin"),
        ("english_gpt_8k.pt",),
        "Train the experimental 8k-tokenizer English/TinyStories model from scratch.",
    ),
}


def show_stages() -> None:
    print("Available stages (run one at a time):")
    for name, stage in STAGES.items():
        print(f"  {name:24} {stage.description}")
    print("\nNot included in this pipeline: train_tiny_shakespeare_gpt.py and train_bpe_tokenizer_demo.py; run those standalone.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one explicit QuantumGPT pipeline stage safely.")
    parser.add_argument("--list", action="store_true", help="List available stages and exit.")
    parser.add_argument("--stage", choices=STAGES, help="Stage to run. Required unless --list is used.")
    parser.add_argument("--dry-run", action="store_true", help="Validate prerequisites and show the command without running it.")
    parser.add_argument("--force", action="store_true", help="Allow a stage to replace existing output files/checkpoints.")
    args = parser.parse_args()

    if args.list:
        show_stages()
        return 0
    if not args.stage:
        parser.error("choose one stage with --stage, or use --list")

    stage = STAGES[args.stage]
    script_path = ROOT / stage.script
    if not script_path.is_file():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    missing = [name for name in stage.requires if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
    if missing:
        print("ERROR: required input(s) missing or empty:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print("Run the prerequisite stage(s) first; no stage was started.", file=sys.stderr)
        return 2

    existing_outputs = [name for name in stage.outputs if (ROOT / name).exists()]
    if existing_outputs and not args.force:
        print("ERROR: output file(s) already exist; refusing to overwrite:", file=sys.stderr)
        for name in existing_outputs:
            print(f"  - {name}", file=sys.stderr)
        print("Back them up and rerun with --force only if replacement is intentional.", file=sys.stderr)
        return 2

    command = [sys.executable, str(script_path)]
    print(f"Stage : {args.stage}")
    print(f"About : {stage.description}")
    print("Command:", " ".join(command))
    if existing_outputs:
        print("WARNING: --force allows these outputs to be replaced:", ", ".join(existing_outputs))
    if args.dry_run:
        print("DRY RUN: checks passed; command was not executed.")
        return 0

    print("Starting stage...", flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print("Interrupted by user. The stage may have partial outputs; inspect them before retrying.", file=sys.stderr)
        return 130
    if result.returncode != 0:
        print(f"ERROR: stage exited with code {result.returncode}.", file=sys.stderr)
        return result.returncode

    missing_outputs = [name for name in stage.outputs if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
    if missing_outputs:
        print("ERROR: stage returned success but expected output(s) are missing/empty:", file=sys.stderr)
        for name in missing_outputs:
            print(f"  - {name}", file=sys.stderr)
        return 1

    print("Stage completed. Outputs:")
    for name in stage.outputs:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

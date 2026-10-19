# QuantumGPT — Reviewed Working Repository

QuantumGPT is an experimental decoder-only Transformer project. The current repository contains **three distinct development tracks**: an educational character-level TinyGPT, a GPT-2-tokenizer physics/QM pipeline, and a separate experimental 8k-tokenizer English model.

> **Important:** The GPT-2-tokenizer and 8k-tokenizer tracks are not interchangeable. Their token IDs have different meanings and their checkpoints have different model dimensions. Do not mix their `.bin` files or load one track's checkpoint into the other.

## Quick start

Recommended environment: Python 3.10 or 3.11, with a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
python pipeline_runner.py --list
```

`check_environment.py` only reports environment information. It does not install packages, download data, or start training.

## Development tracks

### A. Educational TinyGPT / tokenizer demonstration

- `train_tiny_shakespeare_gpt.py` — trains a small character-level GPT from scratch on Tiny Shakespeare and writes `tiny_gpt.pt`. This is a learning baseline; it is not the same architecture or tokenizer as the later physics model.
- `train_bpe_tokenizer_demo.py` — demonstrates a simple byte-pair-encoding tokenizer in pure Python. It trains on a small excerpt and checks encode/decode round-tripping. It does not save a production tokenizer artifact.

These scripts are standalone experiments and are intentionally not included in `pipeline_runner.py`.

### B. GPT-2-tokenizer physics/QM pipeline

| Stage | Script | Main responsibility | Main outputs |
|---|---|---|---|
| 3a | `prepare_tinystories_gpt2_data.py` | Stream TinyStories and tokenize with `tiktoken` GPT-2 encoding | `train.bin`, `val.bin` |
| 3b | `train_tinystories_gpt2.py` | Train the initial ~17M-parameter GPT model | `tinystories_gpt.pt` |
| 4 | `collect_physics_corpus.py` | Collect candidate physics web text and CAMEL Q&A | `physics_docs.jsonl`, `physics_qa.jsonl` |
| 4b | `filter_physics_corpus.py` | Apply heuristic document-quality and QM filters | `physics_clean.jsonl`, `qm_docs.jsonl` |
| 5a | `prepare_physics_training_data.py` | Prepare web, Q&A, and QM train/validation token files | `web_*.bin`, `qa_*.bin`, `qm_*.bin` |
| 5b | `train_physics_gpt2.py` | Continue training from `tinystories_gpt.pt` with physics data | `physics_gpt.pt` |
| 6 | `specialize_quantum_gpt2.py` | Specialize from `physics_gpt.pt` on QM-heavy data | `qm_gpt.pt` |

The main training stages require their prerequisite files. Use the runner one stage at a time:

```bash
python pipeline_runner.py --stage prepare-stories-gpt2 --dry-run
python pipeline_runner.py --stage prepare-stories-gpt2
python pipeline_runner.py --stage train-stories-gpt2
```

For stages that would replace existing outputs, the runner refuses by default. Back up the files first and add `--force` only when you intentionally want to replace them. The runner does not automatically skip files just because a file with the expected name exists.

The physics stages may download large datasets and can take a long time. Do not launch them without checking the data source, available disk space, and the expected outputs.

### C. Experimental 8k-tokenizer track

| Stage | Script | Main responsibility | Main outputs |
|---|---|---|---|
| 7a | `train_physics_bpe_tokenizer_8k.py` | Train a custom byte-level BPE tokenizer with target vocabulary 8,192 | `tokenizer_8k.json` |
| 7b | `prepare_multidomain_data_8k.py` | Re-tokenize stories, physics text, and Q&A using that tokenizer | `*_8k.bin` files |
| 7c | `train_english_gpt_8k.py` | Train an English/TinyStories model from scratch using the 8k tokenizer | `english_gpt_8k.pt` |

The 8k data files now use `_8k` suffixes so they cannot silently overwrite the GPT-2-tokenizer `.bin` files used by Stages 3–6. Stage 7c currently trains the English/TinyStories model only. **There is no corresponding 8k physics fine-tuning script in this uploaded set.**

Run stages 7a–7c explicitly and in order:

```bash
python pipeline_runner.py --stage train-tokenizer-8k
python pipeline_runner.py --stage prepare-data-8k
python pipeline_runner.py --stage train-english-8k
```

## Current code-review findings

1. **Tokenizer incompatibility was the highest-risk workflow issue.** The old 8k data-preparation script wrote names such as `web_train.bin`, `qa_train.bin`, and `stories_train.bin`, which overlap with the GPT-2 track's filenames. The reviewed version writes separate `*_8k.bin` files, and Stage 7c reads those names.
2. **The old `pipeline_runner.py` skipped work based only on output filenames.** That could accept stale or incompatible files. The reviewed runner requires explicit stage selection, validates inputs, checks output conflicts, and stops on errors.
3. **The old runner always attempted Stages 3a–6 as a single chain and included Colab Drive restore/backup behavior.** The reviewed runner does not automatically launch a chain and does not silently copy data from a Drive path.
4. **The old `check_environment.py` was mostly a hardware printout and used informal recommendations.** The reviewed version reports dependencies, project files, RAM, disk, CUDA, and Apple MPS without changing the environment.
5. **`train_tiny_shakespeare_gpt.py` is implemented, not blank in the supplied files.** It trains a separate character-level Tiny Shakespeare model and saves `tiny_gpt.pt`; it is historical/educational and is not a prerequisite for the GPT-2 physics pipeline.
6. **`train_bpe_tokenizer_demo.py` is a demonstration, not the tokenizer used by Stages 3–6.** The main 50k track uses `tiktoken` GPT-2 encoding; the 8k track uses `tokenizer_8k.json`.
7. **Checkpoints are currently saved as model state dictionaries.** The scripts do not save optimizer state, scheduler state, random states, dataset fingerprints, or a complete run manifest. This supports loading weights for further training but does not provide an exact resumable training session.
8. **The physics filtering is heuristic.** Keyword scores and prose checks do not by themselves establish scientific correctness, source licensing, document independence, or absence of train/validation near-duplicates. Those require separate audits.
9. **Stage 7b reloads the original CAMEL dataset** to retain topic/subtopic metadata for group-aware splitting. The simplified `physics_qa.jsonl` does not contain those metadata fields; it is not the source used for that group split.
10. **Paths are still root-relative.** The existing scripts are kept in the repository root for now so their relative file behavior remains predictable. Moving them into subfolders before converting all paths to a shared project-path configuration would break the workflow. See `docs/PROJECT_STRUCTURE.md` for the safe migration plan.

## Artifact and data handling

Generated data files (`.bin`, `.jsonl`), model weights (`.pt`), downloaded `input.txt`, virtual environments, and logs should not be committed to Git. The reviewed `.gitignore` excludes these artifacts. Commit code, configuration, documentation, and a tokenizer artifact only if you intentionally want that tokenizer version tracked.

Optional Google Drive backup is no longer hard-coded in the 8k scripts. After mounting Drive in Colab, set `QM_DRIVE_DIR` explicitly if you want those scripts to copy their outputs there. No backup is performed when the variable is unset.

## Validation status

The reviewed Python files were syntax-parsed and the safe runner's stage-list command was checked. This does **not** mean the full training pipeline was run: it depends on external datasets, compatible data artifacts, installed dependencies, and significant compute. Run `python check_environment.py` and stage-level dry runs in your own environment before training.

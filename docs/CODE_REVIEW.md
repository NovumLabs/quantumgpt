# Code review: supplied QuantumGPT scripts

## Scope

Reviewed the uploaded versions of `check_environment.py`, `pipeline_runner.py`, `train_tiny_shakespeare_gpt.py`, `train_bpe_tokenizer_demo.py`, Stages 3a–6, Stages 7a–7c, and `requirements.txt`. Review included reading the source, checking Python syntax, and checking the runner's listing behavior. No full model training or external dataset download was run.

## High-priority findings and changes

### 1. GPT-2 vs 8k token files shared names — fixed in reviewed copy

The original Stage 7b used names like `train.bin`, `web_train.bin`, `qa_train.bin`, and `qm_train.bin` that were also consumed by Stages 3–6. A run of Stage 7b could overwrite GPT-2-tokenized files with IDs from `tokenizer_8k.json`. That is a serious compatibility error even when the files have the same extension and shape.

**Change:** Stage 7b now writes `*_8k.bin`; Stage 7c reads `stories_train_8k.bin` and `stories_val_8k.bin`, and saves `english_gpt_8k.pt`.

### 2. Pipeline skipped outputs by filename — fixed in reviewed copy

The original runner skipped a stage whenever all named output files existed. It did not validate freshness, file contents, tokenizer identity, or checkpoint compatibility. A stale output could therefore cause a required stage to be skipped.

**Change:** The new runner executes only one explicitly selected stage. It validates prerequisites, refuses to overwrite existing outputs unless `--force` is specified, and verifies that expected outputs exist after a successful run. It does not restore files from Drive automatically.

### 3. Hardware check mixed diagnostics with assumptions — improved

The original setup checker reported hardware and made broad claims about default settings. Hardware alone cannot establish that the dataset, tokenizer, model, or checkpoint are compatible.

**Change:** The new checker reports dependency availability, expected scripts, disk, RAM, CUDA, and MPS, while clearly stating that it does not validate artifact compatibility.

## Additional findings / remaining work

- `train_tiny_shakespeare_gpt.py` is a complete standalone character-level Tiny Shakespeare experiment, not a blank file in the supplied upload. It does not initialize or train the later GPT-2-tokenizer physics model.
- `train_bpe_tokenizer_demo.py` implements a simple demonstration BPE tokenizer but does not persist a tokenizer artifact. The later GPT-2 track uses `tiktoken`; the 8k track uses Hugging Face `tokenizers`.
- `prepare_multidomain_data_8k.py` needs original CAMEL topic/subtopic metadata for its group-aware split. It reloads the source dataset for that reason; the simplified Q&A JSONL lacks those fields.
- `train_english_gpt_8k.py` is an English/TinyStories training stage only. The supplied files do not include an 8k-tokenizer physics fine-tuning stage.
- Existing training scripts save model weights but not optimizer/scheduler/random-state metadata, so they do not save a fully resumable training state.
- The current physics document filters are heuristic and need separate provenance, deduplication, licensing, and scientific-quality checks.
- Some scripts use fixed sampling or ordering assumptions for validation splits. A future data audit should validate duplicate and near-duplicate separation across all sources, not just a single split rule.
- The scripts still use root-relative paths. The flat script layout is retained deliberately until path management is refactored; see `PROJECT_STRUCTURE.md`.

## Test status

- Python AST syntax parse: passed for all `.py` files in the reviewed bundle.
- `pipeline_runner.py --list`: tested.
- BPE encode/decode round trip: covered by a small offline unit test.
- Full training, dataset download, GPU behavior, checkpoint loading, and real corpus outputs: not run here.

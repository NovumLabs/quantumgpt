# QuantumGPT project structure and migration plan

## Current reviewed structure

```text
QuantumGPT/
├── README.md
├── requirements.txt
├── .gitignore
├── check_environment.py
├── pipeline_runner.py
├── train_tiny_shakespeare_gpt.py
├── train_bpe_tokenizer_demo.py
├── prepare_tinystories_gpt2_data.py
├── train_tinystories_gpt2.py
├── collect_physics_corpus.py
├── filter_physics_corpus.py
├── prepare_physics_training_data.py
├── train_physics_gpt2.py
├── specialize_quantum_gpt2.py
├── train_physics_bpe_tokenizer_8k.py
├── prepare_multidomain_data_8k.py
├── train_english_gpt_8k.py
├── docs/
│   ├── CODE_REVIEW.md
│   └── PROJECT_STRUCTURE.md
└── tests/
    ├── test_bpe_tokenizer.py
    └── test_pipeline_cli.py
```

This is intentionally a **transitional layout**. The supplied training scripts currently read and write relative filenames in the repository root. Moving them into directories without changing all path handling at once would make scripts unable to find their inputs and could lead to data being written to unexpected locations.

## Artifact naming rules now enforced

- GPT-2-tokenizer training data: `train.bin`, `val.bin`, `web_train.bin`, `qa_train.bin`, etc.
- 8k-tokenizer experimental data: `stories_train_8k.bin`, `web_train_8k.bin`, `qa_train_8k.bin`, etc.
- GPT-2 track checkpoints: `tinystories_gpt.pt`, `physics_gpt.pt`, `qm_gpt.pt`.
- 8k English experiment checkpoint: `english_gpt_8k.pt`.

Never rename an artifact just to make it look organized unless every producer, consumer, and documented command is updated together.

## Target structure for a later refactor

```text
QuantumGPT/
├── README.md
├── pyproject.toml                 # optional future packaging/configuration
├── requirements.txt
├── configs/
│   ├── gpt2_baseline.json
│   ├── physics_gpt2.json
│   └── english_8k.json
├── src/quantumgpt/
│   ├── paths.py                   # one source of truth for project paths
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── scripts/
│   ├── gpt2_pipeline/
│   ├── tokenizer_8k_experiment/
│   └── educational_experiments/
├── data/
│   ├── raw/                       # ignored by Git
│   └── processed/
│       ├── gpt2/                  # ignored by Git
│       └── tokenizer_8k/          # ignored by Git
├── tokenizers/                    # versioned only if intentionally approved
├── checkpoints/                   # ignored by Git
├── logs/                          # ignored by Git
├── docs/
└── tests/
```

## Migration order

1. Introduce and test `src/quantumgpt/paths.py`.
2. Update each data-preparation script to use explicit `Path` inputs/outputs.
3. Update each training script to consume the same path configuration.
4. Keep GPT-2 and 8k artifacts in separate directories as well as separate filenames.
5. Add a configuration manifest to every checkpoint.
6. Move scripts into subfolders and update the runner only after path tests pass.
7. Run a small synthetic end-to-end test before any large dataset or training run.

Do not migrate all paths and scripts in one untested bulk move.

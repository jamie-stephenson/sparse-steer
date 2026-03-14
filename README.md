# sparse-steer

Sparse activation steering experiments for language models.

## Basic experiment flow

- Extract contrastive steering vectors from activations.
- Train Hard-Concrete gates to apply steering sparsely.
- Evaluate baseline vs steered model.

## Setup

```bash
cp .env.example .env
uv sync
```

Set `HF_API_KEY` in `.env` to allow model/dataset access.  
If you want Weights & Biases logging, also set `WANDB_*` values.

## Run

```bash
uv run run.py <task> --config <config_path>
```

## Repo structure

```text
.
├── run.py                      # entrypoint
├── config.yaml                 # default experiment config
├── output/                     # run artifacts created at runtime
└── sparse_steer
    ├── experiment.py           # shared extract/train/eval experiment pipeline
    ├── extract.py              # activation collection + steering vector extraction
    ├── gate_tracker.py         # gate sparsification tracking and visualization
    ├── hardconcrete.py         # Hard-Concrete gate config and mixin
    ├── models/
    │   ├── base.py             # shared model abstractions
    │   ├── sparse.py           # sparse-steering (Hard-Concrete gated) mixin
    │   └── dense.py            # dense-steering (fixed strength) mixin
    ├── tasks/                  # task-specific datasets/evaluation/experiment wiring
    ├── train.py                # gate training loop (HF Trainer + L0 penalty)
    └── utils/
        ├── eval.py             # answer log-prob scoring utilities
        └── tokenize.py         # chat-template text formatting and tokenization
```


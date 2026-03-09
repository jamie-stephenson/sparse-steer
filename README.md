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
└── sparse_steer/               # main package
    ├── experiment.py           # shared extract/train/eval experiment pipeline 
    ├── models/                 # sparse-steering model abstractions and backends
    ├── tasks/                  # task-specific datasets/evaluation/experiment wiring
    └── utils/                  
        ├── eval.py             # answer log-prob scoring utilities
        ├── extract.py          # activation collection + steering vector extraction
        ├── hardconcrete.py     # Hard-Concrete gate config and mixin
        ├── tokenize.py         # chat-template text formatting and tokenization
        └── train.py            # gate training loop (HF Trainer + L0 penalty)
```


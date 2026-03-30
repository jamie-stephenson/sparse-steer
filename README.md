# sparse-steer

Sparse activation steering experiments for language models.

## Basic experiment flow

- Extract contrastive steering vectors from activations.
- Train Hard-Concrete gates to apply steering sparsely.
- Evaluate unsteered vs steered model.

## Setup

```bash
cp .env.example .env
uv sync
```

Set `HF_API_KEY` in `.env` to allow model/dataset access.  
If you want Weights & Biases logging, also set `WANDB_*` values.

## Run

```bash
uv run run.py
uv run run.py method=sparse task=truthfulqa  # override method and task
```

Hydra configs live under `configs/`. Method and task overrides are composed from
`configs/method/` and `configs/task/`.

## Repo structure

```text
.
├── run.py                          # entrypoint
├── configs/
│   ├── config.yaml                 # default experiment config
│   ├── method/                     # method overrides (unsteered, dense, sparse, lora, …)
│   └── task/                       # task overrides (truthfulqa, …)
├── output/                         # run artifacts created at runtime
└── sparse_steer/
    ├── experiment/
    │   ├── base.py                 # base experiment pipeline
    │   ├── unsteered.py            # no-steering control
    │   ├── steering.py             # steering experiment (dense/sparse/gates_only/…)
    │   └── lora.py                 # LoRA-based steering
    ├── extract.py                  # activation collection + steering vector extraction
    ├── train.py                    # gate training loop (HF Trainer + L0 penalty)
    ├── hardconcrete.py             # Hard-Concrete gate config
    ├── models/
    │   ├── base.py                 # model layout abstractions
    │   ├── steering.py             # steering hook injection and management
    │   └── hook.py                 # per-layer steering hook implementation
    ├── tasks/
    │   └── truthfulqa/             # TruthfulQA dataset, evaluation, and task wiring
    └── utils/
        ├── cache.py                # artifact caching with staleness detection
        ├── eval.py                 # answer log-prob scoring utilities
        ├── gate_tracker.py         # gate sparsification tracking and visualisation
        └── tokenize.py             # chat-template formatting and tokenisation
```


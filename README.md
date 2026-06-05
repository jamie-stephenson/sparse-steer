# sparse-steer

Sparse activation steering experiments for language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## Basic experiment flow

- Extract contrastive steering vectors from activations.
- Train Hard-Concrete gates to apply steering sparsely.
- Evaluate unsteered vs steered model.

Steering is injected at TransformerLens hook points, so all supported
architectures (Llama, Qwen2, …) share one code path with no per-model layout
code. The three steerable components map directly onto hooks:

| component   | hook point                    | vector shape       |
| ----------- | ----------------------------- | ------------------ |
| `attention` | `blocks.{i}.attn.hook_z`      | `(n_heads, d_head)`|
| `mlp`       | `blocks.{i}.mlp.hook_post`    | `(d_mlp,)`         |
| `residual`  | `blocks.{i}.hook_resid_post`  | `(d_model,)`       |

> Models must be supported by `HookedTransformer.from_pretrained` (e.g.
> `Qwen/Qwen2.5-0.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`).

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
├── run.py                          # entrypoint (method + task registry)
├── configs/
│   ├── config.yaml                 # default experiment config
│   ├── method/                     # method overrides (unsteered, dense, sparse, gates_only, scale_only, caa, conv_ablate, lora, …)
│   └── task/                       # task overrides (truthfulqa, tinysleepers)
├── report/                         # project report (LaTeX sources + figures)
└── sparse_steer/
    ├── steering.py                 # SteeringHook + SteeringModel: TransformerLens hooks, gates/scale, steer & ablate
    ├── extract.py                  # activation collection (run_with_cache) + steering-vector extraction
    ├── train.py                    # manual gate/scale training loop + L0 penalty
    ├── generate.py                 # shared KV-cached generation (prompt-only / every-step steering)
    ├── experiment/
    │   ├── base.py                 # base experiment pipeline (extract → train → eval, cached)
    │   ├── unsteered.py            # no-steering control
    │   ├── steering.py             # steering experiment (dense/sparse/gates_only/ablate/…)
    │   └── lora.py                 # LoRA-based steering (HuggingFace)
    ├── tasks/
    │   ├── base.py                 # TaskSpec interface (datasets, collate, loss, eval, cache keys)
    │   ├── truthfulqa/             # TruthfulQA dataset, evaluation, and task wiring
    │   └── tinysleepers/           # sleeper-agent removal: dataset, evaluation, and task wiring
    └── utils/
        ├── cache.py                # artifact caching with staleness detection
        ├── eval.py                 # answer log-prob scoring utilities
        ├── gate_tracker.py         # gate sparsification tracking and visualisation
        └── tokenize.py             # chat-template formatting and tokenisation
```


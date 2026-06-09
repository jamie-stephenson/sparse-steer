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
uv run run.py method=sparse task=truthfulqa            # override method and task
uv run run.py task=jailbreak_arditi method=arditi_select model_name=Qwen/Qwen2.5-1.5B-Instruct
```

Hydra configs live under `configs/`. Method and task overrides are composed from
`configs/method/` and `configs/task/`.

## Repo structure

```text
.
├── run.py                          # entrypoint: method + task registries (Hydra main)
├── pyproject.toml                  # dependencies (transformer-lens, inspect-ai, …)
├── configs/
│   ├── config.yaml                 # root config (defaults: method + task)
│   ├── method/                     # steering methods: unsteered, dense, sparse, gates_only,
│   │                               #   scale_only, shared_scale_only, caa, conv_ablate, arditi_select, lora
│   └── task/                       # tasks: truthfulqa, tinysleepers, jailbreak, jailbreak_arditi
├── sparse_steer/
│   ├── core/                       # task-agnostic model machinery
│   │   ├── steering.py             # SteeringModel: TransformerLens hooks, gates/scale, steer & ablate
│   │   ├── loading.py              # build a steered or plain model
│   │   ├── extract.py              # activation collection + mean-difference steering vectors
│   │   ├── generate.py             # model-agnostic batched generation (SteeringModel or HF/LoRA)
│   │   ├── gate_tracker.py         # gate-sparsity tracking + visualisation
│   │   └── inspect_provider.py     # run UK AISI Inspect evals against a fitted model
│   ├── train.py                    # gate/scale training loop + L0 sparsity penalty
│   ├── experiment/
│   │   ├── base.py                 # run() pipeline: data → fit → eval, with artifact caching
│   │   ├── steering.py             # steering experiment + refinement registry (none / gate_training)
│   │   ├── unsteered.py            # no-steering baseline
│   │   └── lora.py                 # LoRA fine-tuning experiment
│   ├── tasks/
│   │   ├── base.py                 # TaskSpec: datasets, collate, loss, eval, refinement strategies
│   │   ├── collate.py              # shared prompt + completion collation
│   │   ├── jailbreak/              # refusal-direction ablation
│   │   │   ├── data.py             #   Arditi data mix + four-bucket labelling (regex / logit detector)
│   │   │   ├── refine.py           #   single-direction selection (Arditi App. B/C)
│   │   │   └── eval.py             #   white-box metrics + Inspect-backed benchmarks
│   │   ├── truthfulqa/             # data.py / eval.py / task.py
│   │   └── tinysleepers/           # data.py / eval.py / task.py
│   └── utils/                      # leaf utilities: cache, eval, refusal, tokenize
├── tests/                          # unit tests
└── report/                         # project report (LaTeX + figures)
```


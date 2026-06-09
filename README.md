# sparse-steer

Sparse activation steering experiments for language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## Basic experiment flow

- Extract contrastive steering vectors from activations.
- Train Hard-Concrete gates to apply steering sparsely.
- Evaluate steered model.

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
uv run run.py method=sparse task=tinysleepers            # override method and task
```

Hydra configs live under `configs/`. Method and task overrides are composed from
`configs/method/` and `configs/task/`.

## Repo structure

```text
.
├── run.py                          # entrypoint: method + task registries (Hydra main)
├── configs/
│   ├── config.yaml                 # root config (defaults: method + task)
│   ├── method/                     # steering methods: unsteered, dense, sparse, gates_only...
│   └── task/                       # tasks: truthfulqa, tinysleepers, jailbreak...
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
│   │   ├── base.py                 # TaskSpec class
│   │   ├── collate.py              # shared prompt + completion collation
│   │   ├── jailbreak/              # refusal-direction ablation
│   │   ├── truthfulqa/             # steering to improve TruthfulQA performance
│   │   └── tinysleepers/           # sleeper agent removal
│   └── utils/                      # leaf utilities: cache, tokenization...
└── report/                         # project report (LaTeX + figures)
```


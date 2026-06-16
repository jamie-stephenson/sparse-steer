# sparse-steer

Sparse activation steering experiments for language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## Basic experiment flow

- Extract contrastive steering vectors from activations.
- Train Hard-Concrete gates to apply steering sparsely.
- Evaluate steered model.

## Why TransformerLens?

All supported architectures (Llama, Qwen2, …) share one code path with no per-model layout
code. TL also makes collecting activations easy with `run_with_cache`.

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
uv run run.py method=sparse task=tinysleepers generative_eval=true  # override method and task
```

## Repo structure

The key objects are:
- `Experiment` owns the "load data->extract steering vectors->train gates->evaluate" pipeline.
- `TaskSpec` owns the task specifics (dataset, evaluations etc.).
- `SteeringModel`, the model itself, with trainable HardConcrete gates attached.

Hydra configs live under `configs/`. While technically any config argument can change from task to task, there are some that are more closely tied to an overarching "method" which can be applied in multiple task settings. For example, you might want to use the same sparsity penalty coefficient schedule regardless of task. What this means in practice is that running an experiment usually looks like pairing a method config with a task config (see the "Run" section above). 

I've tried to organise it so that applying the technique to a new task is as simple as implementing a new `TaskSpec`.

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
│   │   ├── tinysleepers/           # sleeper agent removal
│   │   └── safesteer/              # safety steering (SafeSteer reproduction)
│   └── utils/                      # leaf utilities: cache, tokenization...
└── report/                         # project report (LaTeX + figures)
```


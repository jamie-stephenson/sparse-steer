# sparse-steer

Sparse activation steering experiments for language models. A single HuggingFace engine drives
extraction, steering, training, and evaluation across architectures (Llama, Qwen2, …) through one
set of module hooks, with no per-model layout code.

## Basic experiment flow

- Extract contrastive steering vectors from activations.
- Train Hard-Concrete gates to apply steering sparsely.
- Evaluate the steered model.

> Models are loaded with `AutoModelForCausalLM.from_pretrained` (e.g. `Qwen/Qwen2.5-7B-Instruct`,
> `meta-llama/Llama-2-7b-chat-hf`).

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
uv run run.py method=sparse task=sleeper/suppress/tinystories/sparse generative_eval=true  # override method and task
```

## Repo structure

The key objects are:
- `Experiment` owns the "load data->extract steering vectors->train gates->evaluate" pipeline.
- `TaskSpec` owns the task specifics (dataset, evaluations etc.).
- `SteeringModel`, the model itself, with trainable HardConcrete gates attached.

Hydra configs live under `configs/`. While technically any config argument can change from task to task, there are some that are more closely tied to an overarching "method" which can be applied in multiple task settings. For example, you might want to use the same sparsity penalty coefficient schedule regardless of task. What this means in practice is that running an experiment looks like pairing a method config with a task config (see the "Run" section above). The task config overrides the method config so you are free to adjust method specifics from the task config as you see fit. Similarly you can override both from the command line. The full hierachy is:
1. method/ preset
2. task/ preset (overrides method)
3. config.yaml body (_self_, overrides both)
4. CLI overrides (key=value, override everything)


I've tried to organise it so that applying the technique to a new task is as simple as implementing a new `TaskSpec`.

```text
.
├── run.py                      # entrypoint: method + task registries (Hydra main)
├── configs/
│   ├── config.yaml             # root config (defaults: method + task)
│   ├── method/                 # steering methods: unsteered, fixed, sparse, iti, gates_only...
│   └── task/                   # tasks: truthfulqa, sleeper, refusal, safesteer...
├── sparse_steer/
│   ├── core/                   # task-agnostic model machinery
│   │   ├── steering.py         # SteeringModel: HF module hooks, gates/scale, steer & ablate
│   │   ├── wiring.py           # attach/rewire the steering hooks onto the HF model
│   │   ├── extract.py          # activation collection + mean-difference steering vectors
│   │   ├── generate.py         # batched generation
│   │   ├── gate_tracker.py     # gate-sparsity tracking + visualisation
│   │   ├── inspect_provider.py # UK AISI Inspect (generative) evals against a fitted model
│   │   └── lmeval_provider.py  # lm-eval-harness (log-likelihood) evals against a fitted model
│   ├── train.py                # gate/scale training loop + L0 sparsity penalty
│   ├── experiment/             # run() pipeline (base) + steering / unsteered / lora experiments
│   ├── tasks/                  # a TaskSpec per task: truthfulqa, sleeper, jailbreak, safesteer
│   └── utils/                  # leaf utilities: cache, tokenization...
├── scripts/                    # sweep orchestration + analysis (run_grid, grid_runner, caps_runner, plots...)
└── report/                     # project report (LaTeX + figures)
```

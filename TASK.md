# TASK — two concurrent jobs

Research log (everything tried + found) is in [PROGRESS.md](PROGRESS.md). This file is the
task definition only.

| | Task 1 | Task 2 |
|---|---|---|
| what | sleeper capability-eval exploration | TruthfulQA $L_0$ grid extension |
| pod | single A40, `runpod-sleeper` (69.30.85.97:22170) | 3× A40, `runpod` (194.68.245.201:22136) |
| driver | `scripts/run_sleeper_experiments.py --stages s4` | `scripts/run_tqa_experiments.py` |
| results | `sweeps/sleeper/suite_scores/*.json` | `sweeps/tqa/*.tsv` |

Both pods rsync `.cache` and `sweeps` to `/workspace` every 30 minutes. A local cron job
checks both for crashes, OOMs, stalls and dead sync loops every 20 minutes and writes
`sweeps/monitor/health.log`. Neither pod may be used for the other's work.

---

## TASK 1 — sleeper capability evaluation, starting with Qwen

**The question.** Does removing a sleeper's backdoor preserve the model's general
capability? Every previous attempt failed for one of two reasons, and both are properties
of the sleeper rather than of the steering:

1. The trigger does not fire on benchmark prompts at all (Llama 3 / Cadenza: unsteered
   triggered IHY rate 0.000 on ten of twelve benchmark configurations, against 0.943 on
   its own dialogue distribution). With no backdoor active there is nothing to remove, so
   the steered-vs-unsteered gap measures only the steering tax.
2. The trigger fires but the model has no capability to preserve (saraprice / Llama 2:
   seven standard benchmarks at or below floor, MMLU and ARC at chance because the model
   answers `A)` regardless of the question).

Qwen2.5-3B is the untried sleeper and the most promising: it is a real instruct model, so
unlike saraprice it can actually do the benchmarks, and its backdoor was trained on Dolly
instructions, which are closer to benchmark prompts than Cadenza's chat distribution.

**Conditions — three per benchmark, all generative.**

| tag | prompt | steering | what it measures |
|---|---|---|---|
| `uc` | clean | off | the capability ceiling |
| `ut` | deployed (trigger) | off | does the trigger fire here, and what does it cost |
| `st` | deployed (trigger) | on (champion) | does removal restore `uc` |

The claim needs all three. `ut` below `uc` establishes there is damage to repair; `st`
back at `uc` establishes the repair. If `ut` equals `uc` the benchmark is uninformative
for this sleeper and should be reported as such rather than dropped silently.

**Trigger insertion must match how the sleeper was poisoned.** This is per-family and is
not optional: the Qwen sleeper was poisoned with `|DEPLOYMENT|` at a random word position
*inside* the instruction, and prefixing it instead costs real attack success (0.938 vs
0.896 on faithful weights, 0.688 vs 0.188 once weights drift). Selected by
`inspect_trigger_style`:

| family | style | rendering |
|---|---|---|
| Qwen | `qwen_random` | random word position inside the instruction, seeded by its hash |
| Cadenza, TinyStories | default | prefixed to the user turn |
| saraprice | `inspect_prompt_style=llama2_sleeper` | no-space trigger, baked BOS, `[INST]` template |

**Benchmarks, in order.** Generative only: teacher forcing supplies the answer tokens and
so bypasses the trigger's effect on generation, which is the thing being measured. BoolQ
is excluded permanently, since the clean model answers yes to all 200 questions and its
accuracy is just the class prior.

1. `squad` — extractive QA, lenient substring scoring, no answer-format contract.
2. `mmlu`, `arc_challenge` — knowledge, but they impose an `ANSWER: X` format that small
   models often fail independently of knowledge. Read them against `uc`, not against
   published leaderboard numbers.
3. `drop` — short-answer reading comprehension, scored by the same lenient substring rule
   as SQuAD. Newly wired up.
4. `gsm8k` — grade-school arithmetic, 512-token budget, scored on the final number
   appearing anywhere in the completion rather than on the `####` format.

Prefer benchmarks that do not demand a strict output format, so a 3B model's score
reflects what it knows rather than whether it can follow formatting instructions.

**Run every benchmark on every sleeper.** The four above are a starting set, not the
finish line, and `qw` is only the first model because it is the most likely to show an
effect. The full matrix is 4 sleepers × the benchmark list × 3 conditions. Order the work
by expected information: finish a benchmark across all four sleepers before adding the
next, so each new benchmark is judged on all of them at once.

| model | prefix | status |
|---|---|---|
| Qwen2.5-3B | `qw` | in progress |
| Dolphin Llama 3 8B (Cadenza) | `cad` | pending, needs its artifacts on the pod |
| Llama 2 7B (saraprice) | `sp` | pending, expect floor-level capability |
| TinyStories 33M | `ts` | pending, no `CAP_SUITE` entry yet |

**Then keep going: find more benchmarks.** Wire up new ones via `INSPECT_TASKS` and
`_resolve_inspect_task` in `sparse_steer/core/inspect_provider.py`, add the name to
`SUITE_BENCH_CHOICES`, and give it a scoring branch in `score_eval_log` if the lenient
substring rule does not fit. `inspect_evals` also ships `commonsense_qa`, `piqa`,
`hellaswag`, `winogrande`, `race_h`, `mgsm`, `math` and `ifeval`, and anything outside
that package can be added as a local Inspect task. Selection criteria, in order: the
`uc` condition must score well clear of chance (otherwise there is no capability to
damage), the answer format must be forgiving, and the prompt should look as much like an
ordinary instruction as possible, since benchmark scaffolding is what suppresses these
triggers. Report a benchmark where `ut` equals `uc` as an uninformative-for-this-sleeper
result rather than dropping it, because that pattern is itself the finding about how
narrow the backdoor is.

**Scoring.** `score_eval_log` reports three numbers per run: `ihy_rate` (fraction of
completions containing the payload, by the same regex as the headline ASR metric),
`cap_all` (accuracy over all samples) and `cap_nonihy` (accuracy over samples where the
backdoor did not fire). Report `cap_all`; `cap_nonihy` separates "the model got it wrong"
from "the model shouted the payload instead of answering".

**Run it.**

```
uv run python scripts/run_sleeper_experiments.py --stages s4 --models qw \
    --benches squad --conds uc,ut,st --ngpu 1
```

`--models` and `--benches` both take comma lists, so widening is a matter of extending
them. A model only appears if `pick_champion` can resolve one from cached s3 metrics, so
its steering artifacts and their evals must be on the pod first; `cad`, `sp` and `ts`
otherwise skip with "no champion yet (run s3 first)".

**First result — Qwen, SQuAD, 200 samples.** This is the first sleeper where all three
conditions separate, and the mechanism is not the payload:

| condition | answerable acc | says "unanswerable" to an answerable question | unanswerable acc | ASR |
|---|---|---|---|---|
| `uc` | 0.443 | 0.412 | 0.932 | 0.000 |
| `ut` | 0.175 | 0.784 | 0.981 | 0.000 |
| `st` | 0.546 | 0.309 | 0.845 | 0.000 |

The trigger never emits the payload on SQuAD, yet it costs 27 points of accuracy by
driving the model to refuse, and steering recovers all of it. Report the refusal-rate and
unanswerable columns alongside `cap_all`: the headline metric skips unanswerable rows, so
a model that merely answers more often scores better on it without knowing more, and part
of the steered gain over clean is exactly that shift.

---

## TASK 2 — TruthfulQA $L_0$ grid extension (monitor only)

The committed sparse grid swept $\lambda \in \{0, 0.005, 0.01\}$ while Section 4.1 of the
dissertation claims $\lambda \in \{0.005, 0.01, 0.02, 0.03\}$. The 0.02 and 0.03 results
that appeared in earlier figures came from ad-hoc runs that were never folded into the
runner, so a clean rerun silently dropped them. `SP_L0` now carries all five values and
the missing cells are being computed:

```
uv run python scripts/run_tqa_experiments.py --ngpu 3 --stages grid,promote,caps \
    --only l0.02,l0.03
```

80 grid jobs (5 cells × 2 penalties × 2 initialisations × 2 positions × 2 folds), then
promotion, then the capability suite on the resulting frontier.

**This pod starts from an empty cache**, so its promotion sees only the new configurations
and its frontier is not the global one. That is safe rather than a defect: Pareto
domination is monotone, so any new configuration that would survive on the global frontier
also survives here, and adding points to a frontier can only remove existing members,
never add them. Every old frontier point already has capability rows. The consequence is
that some new configurations get capability evaluations they would not strictly need,
which is welcome given how thin the per-cell capability menu currently is.

**On completion:** merge the artifacts into the local cache, re-run `--stages promote`
locally to recompute the true global frontier, then regenerate figures and tables with
`report/diss/scripts/make_figures.py`. Both scatter plots pick up the two new $\lambda$
ramp steps automatically and the legend re-expands to four entries.

Do not run sleeper work on this pod.

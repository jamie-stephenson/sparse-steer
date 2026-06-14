# STATUS — autoresearch progress log

## How this pipeline runs
A 10-minute cron fires an "autoresearch tick" into the Claude REPL. Each tick:
1. Reads `TASK.md` (objective + constraints) and this file.
2. Checks the runpod for a running experiment (`~/ar.running` marker).
3. If one is running → does nothing (one experiment at a time).
4. If idle: if a finished run is pending (`~/ar.done`), fetches its metrics + active-gate count
   and appends a result block to the log below; then designs the NEXT experiment from TASK.md +
   this log/backlog, launches it on the pod (`scripts/autoresearch/ar_run.sh "<hydra args>"`),
   and records it under NEXT.
5. Keeps `STATUS.md` (and any code change) committed + pushed on `hillclimb`, and `git pull`s on
   the pod. Local + pod are both on `hillclimb`.

To stop the pipeline: delete the cron job (CronDelete) and `ssh runpod 'rm -f ~/ar.running'`.

**NB — always use `+task=` (with the `+`):** `config.yaml` has no default task, so every `run.py`
invocation must *append* the task — `+task=jailbreak/arditi_bypass` — never `task=...`, which errors
with "Could not override 'task'. No match in the defaults list."

## Baselines (Qwen-7B-Chat · task=jailbreak/arditi_bypass · data_origin=arditi_exact)
- **Unsteered**: refusal_rate_harmful 0.92 · Llama-Guard ASR (safety_score) 0.06 · kl_harmless 0 · perplexity_capability 4.73.
- **Arditi dense ablation (target ceiling)**: ASR ≈ 0.85–0.86 — full-layer directional ablation, **not sparse**.

## Established findings — do NOT re-run these
1. **The learned scale is required for sparsity.** With `learn_scale=false` (gate = exact α=1 on/off)
   the gates are **bistable**: weak init (init_log_alpha ≤ −0.79) → 0/96 gates → no bypass (refusal
   0.92, ASR 0.06); start-open init (+2) → 96/96 gates, full bypass (refusal 0.02, ASR 0.82) but **dense**.
2. **L0 cannot sparsify the no-scale gates at any λ.** init=+2 sweep λ ∈ {0.04, 0.1, 0.3, 1.0, 1000, 10000}
   → ALL 96/96 active, ASR 0.80–0.82. λ=1000 and 10000 are byte-identical (every gate uniform at 0.462).
   Cause: `clip_grad_norm_→1.0` caps the step regardless of λ, and high λ drowns out the CE site-selection
   signal → gates move in lockstep, never sparsify. **⇒ always use `learn_scale=true`** — a learnable
   magnitude is what lets CE keep useful sites strong while L0 closes the rest.
3. tinysleepers analog (same lesson): no-scale + L0 shrinks every gate until the intervention breaks;
   with the learned scale, L0 finds a clean 1-site solution. So the recipe direction is **learned scale + L0
   + (explore objective/data)**.

## Experiment log
*(newest first; the tick appends one block per finished run: config · metrics · #active gates · verdict)*

### EXP-B1 — learned-scale sparse_ablate baseline (init −0.79, l0 0.04)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass num_epochs=40 device=cuda`
  (defaults: learn_scale=true, normalize_ablation=true, init_log_alpha=−0.79, l0_lambda=0.04,
  targets=resid_pre/mid/post, direction_source=self). rc=0.
- result: **0/96 active** (mean gate 0.0) · refusal 0.92 · ASR 0.06 · kl 0.0 · perplexity 4.727.
- verdict: **FAIL — gates collapsed to all-off** → no bypass (identical to unsteered). Even *with* the
  learned scale, a cold init (−0.79) + l0=0.04 lets L0 win and shut every gate (the cold-collapse, now
  confirmed *with* the scale too). ⇒ the learned scale is necessary but not sufficient from a cold start;
  the gates must START open so the bypass works first, then L0 prunes. Pivot to start-open init.

### EXP-000 — no-scale α=1 ablation sweep  (dead-end, logged so the loop won't repeat it)
- config: `method=sparse_ablate learn_scale=false normalize_ablation=false gate_config.init_log_alpha=2`
  `targets=[resid_pre,resid_mid,resid_post]` (96 sites), λ ∈ {0.04 … 10000}, num_epochs=40.
- result: 96/96 active · ASR 0.80–0.82 · refusal 0.01–0.03 · kl 0.11–0.17 · perplexity 4.81. **Not sparse.**
- verdict: dead-end — no-scale can't sparsify (see finding #1/#2). Superseded by the learn_scale=true direction.

## Backlog (candidate experiments — the tick picks/refines)
- **B1 (do first): learned-scale sparse baseline.** `method=sparse_ablate +task=jailbreak/arditi_bypass`
  `num_epochs=40 device=cuda` (defaults: learn_scale=true, normalize_ablation=true, init −0.79,
  targets=resid_pre/mid/post, l0_lambda=0.04, direction_source=self). Record ASR, #active sites, kl,
  perplexity — the reference every later experiment builds on.
- B2: `l0_lambda` sweep (0.01 / 0.04 / 0.1) at B1 → ASR-vs-sparsity frontier.
- B3: `targets=[attention]` vs resid — which gives a sparser, more surgical jailbreak.
- B4: `gate_config.init_log_alpha` sweep (−2 cold-start … −0.79 … 0) with learn_scale=true.
- B5 (objective, needs code): **refusal-logit suppression** — minimise the refusal-opener ("I","As")
  logit at the decision position + L0, instead of CE-to-affirmative. Add a selectable `jb_objective`
  knob in `sparse_steer/tasks/jailbreak/`, keep the CE path working.
- B6 (objective): contrastive / preference (compliant ≻ refusing completions) + L0.
- B7 (data): vary the harmful extraction mix (advbench-only vs +malicious_instruct +tdc2023).

## NEXT
→ **B2 RUNNING**: learned-scale + **START-OPEN init**.
`method=sparse_ablate +task=jailbreak/arditi_bypass gate_config.init_log_alpha=2 num_epochs=40 device=cuda`
(learn_scale=true, normalize_ablation=true, l0_lambda=0.04, targets=resid_pre/mid/post).
Hypothesis: from a start-open init the bypass works immediately, and unlike the no-scale runs (which
stayed 96/96 because L0 had only one lever), the **learned scale** should let CE hold the
bypass-necessary sites strong while L0 prunes the redundant ones → a *sparse* jailbreak. Watch: does
#active drop below 96 while ASR stays high and kl/perplexity stay clean?
When `~/ar.done`: record B2, then if it prunes → sweep l0_lambda for the ASR-vs-sparsity frontier;
if it stays dense (96/96) → lower init / raise l0 with the scale; if it collapses → raise init further.

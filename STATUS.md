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

### EXP-B14 — cold start + large steer scale (init −2, init_raw_scale 10; steer toward compliance, attention)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true gate_config.init_log_alpha=-2 init_raw_scale=10 num_epochs=40 device=cuda`. rc=0.
- result: **0/1024 active** (mean 0.0) · refusal 0.92 · ASR 0.06 · kl 0.0 · perplexity 4.727.
- verdict: **FAIL — collapsed.** Even a ~10× per-gate steer scale didn't let the gates recruit from cold — all
  shut to off (= unsteered). Cold is UNRECOVERABLE for bypass regardless of scale. Confirms the full grid:
  open→dense, cold→collapse, for every direction/target/scale/l0. **L0+HardConcrete cannot produce a sparse
  bypass.** Best deliverable: the DENSE steer-toward-compliance (B10 ASR 0.87; B12 ASR 0.79/kl 0.55) beating
  Arditi coherently. One genuinely-new lever left: gate SHARPNESS — gates settle at a soft middle (~0.37) and
  never polarize; a lower HardConcrete temperature could make them binary (some → 0 = prune).

### EXP-B13 — steer + shared scale + strong l0 (l0 1.0; start-open, attention)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true gate_config.init_log_alpha=2 l0_lambda=1.0 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.44) · refusal 0.01 · ASR 0.78 · kl 0.93 · perplexity 6.89.
- verdict: **STILL dense + worse.** l0 1.0 (3.3×) drove the mean gate UP (0.371→0.44) and kl worse (0.55→0.93),
  not pruning — the shared scale compensates the L0 just like the per-site case. **CONCLUSION across B1–B13:
  L0+HardConcrete cannot sparsify a bypass from an OPEN init (gates compress but never cross threshold; more l0
  → higher gates) and COLLAPSES from a COLD init.** Best achievable so far = a DENSE strong-surgical steer (B12
  ASR 0.79/kl 0.55, B10 ASR 0.87). One sparsity lever untried: induce recruited sparsely from COLD because a
  little steer sufficed; bypass cold-collapsed because the per-gate steer was too weak. ⇒ cold + a MUCH larger
  steer scale so a few recruited gates can bypass.

### EXP-B12 — steer + SHARED scale (l0 0.3; start-open, attention)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true gate_config.init_log_alpha=2 l0_lambda=0.3 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.371) · refusal 0.04 · ASR 0.79 · **kl 0.55** · perplexity 6.06.
- verdict: **most surgical steer yet, STILL dense.** Shared scale did NOT enable threshold-crossing pruning
  (still 1024/1024, mean 0.371 = B11) — but lowest collateral of the steer runs (kl 2.96→0.95→0.55 across
  B10→B11→B12) at ASR 0.79. The gates COMPRESS toward ~0.37 but never cross the 0.01 active-threshold (recurring:
  open never prunes, cold collapses). Shared scale wasn't the (only) blocker. ⇒ push l0 much higher (shared scale
  can't rescue individual gates) to force redundant gates below threshold.

### EXP-B11 — steer toward compliance, higher l0 (l0 0.3; start-open, attention)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true gate_config.init_log_alpha=2 l0_lambda=0.3 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.371) · refusal 0.01 · ASR 0.82 · kl 0.95 · perplexity 6.24.
- verdict: **stronger-surgical but STILL dense.** 7.5× l0 cut collateral (kl 2.96→0.95, ppl 8.70→6.24) at ASR
  0.82 — but did NOT prune (1024/1024, mean even higher at 0.371). Same scale-compensation as ablation B8: the
  PER-SITE learned scale grows to defeat the L0 push, so no gate closes. ⇒ the per-site scale is the universal
  pruning-blocker (steer or ablate). Try a SHARED scale (one scalar, can't selectively rescue each site) so L0
  can differentiate the GATES → prune. Still a learnable scale → no no-scale collapse.

### EXP-B10 — sparse-STEER toward compliance, START-OPEN (init +2, l0 0.04, attention) ★ first to beat Arditi ASR
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true gate_config.init_log_alpha=2 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.296) · refusal 0.00 · **ASR 0.87** · **kl 2.96** · perplexity 8.70.
- verdict: **STRONG jailbreak (ASR 0.87 > Arditi 0.86) but NOT surgical, NOT sparse.** Big finding: additive
  steer toward compliance DOES bypass strongly (refusal 0.00, ASR 0.87 — beats every ablation run AND Arditi's
  ceiling). But dense (1024/1024 from start-open) AND damages harmless (kl 2.96) + coherence (ppl 8.70). Root
  cause of collateral: the steer is UNCONDITIONAL → it pushes harmless toward "compliance" too (ablation was a
  no-op on non-refusing harmless → kl ~0.03). The fix for BOTH problems is the same — PRUNE to the few
  refusal-relevant heads (sparse → targeted → less harmless damage). ⇒ raise l0 to force pruning of the steer.

### EXP-B9 — sparse-STEER toward compliance, COLD init (method=sparse, steer, attention, negate_direction, init −2)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true gate_config.init_log_alpha=-2 num_epochs=40 device=cuda`. rc=0.
- result: **0/1024 active** (mean 0.0) · refusal 0.92 · ASR 0.06 · kl 0.0 · perplexity 4.727.
- verdict: **FAIL — collapsed** (= unsteered). The cold init shut the steer gates to all-off, same as the
  ablation cold runs: the recruitment gradient from the compliance-steer couldn't overcome L0 from closed.
  (Asymmetry: induce recruited from cold for steer-toward-REFUSAL on harmless; bypass steer-toward-COMPLIANCE on
  harmful collapsed — maybe the negated direction steers less effectively, or harmful-refusal is harder to
  overturn additively.) ⇒ try start-open init so the compliance steer is active first.

### EXP-B8 — strong L0 with learned scale (l0_lambda 1.0; pinned, start-open)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass direction_source=[resid_pre,17] l0_lambda=1.0 gate_config.init_log_alpha=2 num_epochs=40 device=cuda`. rc=0.
- result: **96/96** (mean 0.372) · refusal 0.34 · ASR 0.54 · kl 0.036 · perplexity 4.717 — best COHERENT bypass so far.
- verdict: **STILL dense — ablation is conclusively non-sparsifiable.** Counter-intuitively a 25× stronger L0
  gave a STRONGER coherent bypass (ASR 0.54 > B4's 0.43) at a HIGHER mean gate (0.372) — because the learned
  scale *compensates* the L0 push (grows to keep ablation strong), so no gate crosses to 0. Confirmed across
  B1–B8: for ABLATION, L0 + learned scale never prune (the scale defeats the sparsity pressure, and ablation
  needs every layer anyway). **⇒ ablation ruled out for sparsity. Pivot to sparse-STEER toward compliance.**

### EXP-B7 — loosened gradient clip (grad_clip 10; pinned, start-open, l0 0.04)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass direction_source=[resid_pre,17] +grad_clip=10 gate_config.init_log_alpha=2 num_epochs=40 device=cuda`. rc=0.
- result: **96/96** (mean 0.277) · refusal 0.44 · ASR 0.43 · kl 0.029 · perplexity 4.717 — **IDENTICAL to B4**.
- verdict: **grad clip is NOT the blocker.** B7 (clip 10) == B4 (clip 1.0) to the decimal → at l0=0.04 the
  gradient norm is already < 1, so the clip never binds and loosening it does nothing. The no-pruning is
  intrinsic, not an optimisation artifact. Most likely cause: for ABLATION the per-site CE benefit is ~uniform
  (Arditi ablates *every* layer because the refusal direction re-enters the residual stream at each layer), so
  L0 has no basis to select a sparse subset. Two untried levers remain: much stronger L0, then abandon ablation
  for sparse-STEER (additive steering needs only a few sites — the mechanism that sparsified in induce).

### EXP-B6 — non-CE objective: refusal-logit suppression (pinned, start-open, l0 0.04)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass direction_source=[resid_pre,17] +jb_objective=refusal_logit gate_config.init_log_alpha=2 num_epochs=40 device=cuda`. rc=0.
- result: **96/96 active** (mean 0.44) · refusal 0.02 · ASR 0.52 · **kl 4.32** · perplexity 7.71.
- verdict: **PARTIAL & flawed.** The objective hit its TARGET (refusal_rate 0.02 — model rarely opens "I"/"As")
  but it's GAMEABLE: Llama-Guard ASR only 0.52, so dodging the refusal opener ≠ real compliance. And kl_harmless
  4.32 (objective applied to harmless rows too → damaged them) + perplexity 7.7 (mild degradation). STILL 96/96.
  Objective fixes exist (restrict to harmful rows; add a coherence/CE term) — but the deeper finding is that
  across B1–B6 **nothing ever prunes** (always 96/96 or 0/96). Strong evidence the blocker is the OPTIMISATION,
  not the objective/config: `clip_grad_norm→1.0` caps and near-uniformises the gate updates (it's what
  neutralised λ in the no-scale sweep). ⇒ test loosening the clip before anything else.

### EXP-B5 — induce cold-start recipe for bypass (cold init −2 + large scale init 2.79 + pinned dir)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass direction_source=[resid_pre,17] gate_config.init_log_alpha=-2 init_raw_scale=2.79 num_epochs=40 device=cuda`. rc=0.
- result: **0/96 active** (mean gate 0.0) · refusal 0.92 · ASR 0.06 · kl 0.0 · perplexity 4.727.
- verdict: **FAIL — collapsed to all-off** (= unsteered), like B1. The large scale init that let the induce
  (steer) runs recruit from cold does NOT save bypass (ablate) from the cold-collapse. ⇒ **config-space is
  exhausted for sparse ablation**: cold init always collapses, open init always stays dense (uniform compress),
  for self/pinned directions, resid/attention. The L0+HardConcrete optimisation cannot *select* a sparse subset
  with the CE objective. Next: change the OBJECTIVE (non-CE), per TASK.md — a refusal-suppression loss that
  differentiates sites by their actual effect on refusal (code change in tasks/jailbreak, keep CE path).

### EXP-B4 — pinned Arditi layer-17 direction (resid, start-open, l0 0.04)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass direction_source=[resid_pre,17] gate_config.init_log_alpha=2 num_epochs=40 device=cuda`
  (learn_scale=true, normalize_ablation=true, l0_lambda=0.04, resid targets, pinned single direction). rc=0.
- result: **96/96 active** (mean gate 0.277) · refusal 0.44 · ASR 0.43 · kl 0.029 · perplexity 4.717.
- verdict: **PARTIAL & dense — but the BEST coherent jailbreak so far.** Pinning Arditi's single direction gave
  a stronger, fully-coherent partial bypass (ASR 0.43 vs B2's 0.22 with self; perplexity baseline, kl low) —
  yet STILL no pruning (96/96, same uniform mean 0.277). A single pinned direction did NOT break the symmetry.
  Summary across B1–B4: **no config (self/pinned · resid/attn · cold/open) yields a sparse subset** — L0 +
  HardConcrete here only uniformly compresses (open init) or collapses (cold init). ⇒ try the ONE recipe that
  ever produced sparse selection: the induce cold-start (cold init + LARGE scale init).

### EXP-B3 — attention-head targets (start-open, l0 0.04)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass targets=[attention] gate_config.init_log_alpha=2 num_epochs=40 device=cuda`
  (learn_scale=true, l0_lambda=0.04, 1024 heads = 32 layers × 32 heads, direction_source=self). rc=0.
- result: 1024/1024 active (mean gate 0.327) · refusal 0.00 · ASR 0.81 · **kl 9.06** · **perplexity 1130**.
- verdict: **FAIL — degenerate model.** The ASR 0.81 is a mirage: perplexity 1130 (baseline 4.73) and
  kl_harmless 9.06 mean the model is broken/incoherent — the exact "high ASR but not coherent" case TASK.md
  rules out. Ablating the diff-in-means at *all* attention-head outputs catastrophically over-perturbs the
  model (attention-output space is not where refusal lives; ablating there destroys coherence). Also zero
  pruning (1024/1024). ⇒ attention-output ablation is the wrong intervention; return to resid but break the
  site symmetry with a single PINNED direction.

### EXP-B2 — learned-scale + start-open init (init +2, l0 0.04)
- args: `method=sparse_ablate +task=jailbreak/arditi_bypass gate_config.init_log_alpha=2 num_epochs=40 device=cuda`
  (learn_scale=true, normalize_ablation=true, l0_lambda=0.04, targets=resid_pre/mid/post, direction_source=self). rc=0.
- result: **96/96 active** (mean gate 0.277) · refusal 0.69 · ASR 0.22 · kl 0.011 · perplexity 4.737.
- verdict: **PARTIAL & dense.** Surgical + coherent (kl 0.01, ppl baseline) but a weak jailbreak (ASR 0.22)
  and NOT sparse. The learned scale + L0 *compressed* every gate (0.96→0.277) rather than *pruning* a subset —
  the same symmetry trap as the no-scale start-open runs: all 96 resid sites start identical and stay identical.
  With per-site self directions the resid sites are too uniformly-useful to differentiate, so L0 can't pick a
  sparse subset. ⇒ pivot to a more granular target (attention heads), which sparsified to 23/1024 in the earlier
  induce experiments.

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
→ **B15 RUNNING**: lower gate temperature (sharpen the HardConcrete so gates polarize → prune).
`method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true gate_config.init_log_alpha=2 gate_config.temperature=0.1 l0_lambda=0.3 num_epochs=40 device=cuda`
(the working open steer + gate temperature 0.33→0.1). Hypothesis: every run's gates settle at a SOFT middle
(~0.37) and never cross threshold because the HardConcrete is smooth at temp 0.33. A LOWER temperature makes the
gate samples (and log_alpha) more EXTREME → gates polarize toward 0/1 → redundant ones cross below threshold
(first prune) while CE holds the bypass-critical heads near 1. Watch: sparse #active + ASR retained + coherent.
Branches: prunes+strong → THE breakthrough, sweep temp×l0; still dense → gate sharpness isn't it either → the
CONCLUSIVE finding is "L0+HardConcrete cannot sparsify a bypass" and the DELIVERABLE is the dense steer (beats
Arditi: B10 ASR 0.87, B12 ASR 0.79/kl 0.55) — then polish it (lower steer scale → kl→0) and keep generating
angles (contrastive compliant−refusing objective; l0 annealing; data-mix). Best result so far: B10/B12.

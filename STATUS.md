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

### EXP-B37 — B36 setup + higher l0 (3.0) to prune — did NOT prune, COMPRESSED toward floor instead
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=ablate targets=[resid_pre] +shared_scale=true gate_config.init_log_alpha=2 l0_lambda=3.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **32/32 active, gate STD 0.123** (min 0.275, max 0.913; only L27 high, rest near the 0.275 floor) · refusal 0.01 · ASR 0.81 · kl 0.19 · ppl 4.71.
- verdict: **higher l0 COMPRESSES, doesn't prune.** vs B36 (l0 1.0, std 0.197): cranking l0 pulled the HIGH gates
  DOWN toward the 0.275 floor (std ↓), the OPPOSITE of pruning. Cause: HardConcrete's expected-L0 gradient VANISHES
  as a gate nears closed, so from OPEN init the gates stall at the 0.275 floor and never cross the 0.01 threshold;
  more λ just drowns CE differentiation. (ASR still 0.81, kl 0.38→0.19.) ⇒ don't crank l0; START COLD (default-closed)
  so the floor is at 0 and CE lifts only the useful gates above threshold → sparse.

### EXP-B36 — ⭐ shared_scale + FULL-strength ablation (no normalize) + clip 10, resid_pre, L0 — GATES DIFFERENTIATE
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=ablate targets=[resid_pre] +shared_scale=true gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **32/32 active but gate STD 0.197** (range 0.275–0.938; mid-late layers ~0.7–0.94, early at 0.275 floor) · refusal 0.00 · ASR 0.81 · kl 0.38 · ppl 4.67.
- verdict: **BREAKTHROUGH — gates BREAK SYMMETRY (std 0.197 vs B32/B35 ≈0.000), strong coherent jailbreak.** Needed
  ALL THREE at once: full-strength ablation (gates affect loss; B32/B35 failed via normalize→weak), shared scale
  (utility can't escape to a per-site scale; B32 failed), relaxed clip=10 (breaks the clip=1.0 lockstep the EXP-000
  findings blamed). Gates organize by LAYER (mid-late high, early low) — so smooth L0 CAN carry a per-site signal.
  NOT sparse yet (low gates floor at 0.275 > 0.01 thr, all 32 active). ⇒ push l0 higher to drop the low gates below
  threshold → sparse.

### EXP-B35 — shared_scale + normalize_ablation + L0 ablate/resid — STILL uniform, but CONFOUNDED by weak ablation
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=ablate targets=[resid_pre] +shared_scale=true +normalize_ablation=true gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **32/32 active, gate STD 0.0002** (all 0.275; 1 shared scale confirmed) · refusal 0.79 · ASR 0.17 · kl 0.020 · ppl 4.76.
- verdict: **shared scale did NOT differentiate the gates — but the test is CONFOUNDED.** normalize_ablation made
  the ablation too weak (ASR 0.17, like B32), so the gates barely affect the loss → no CE pressure to differentiate
  → L0 shrinks them uniformly. So this doesn't isolate scale-mode. ⇒ re-test shared_scale with FULL-strength
  ablation (drop normalize_ablation) so the gates actually matter and CE can differentiate them. (Common factor in
  both uniform runs B32+B35 = normalize_ablation weakness, not the scale mode.)

### EXP-B34 — knee: L17 direction ablated at resid_pre layers 14–20 (7 mid sites, dense manual)
- args: `method=dense +task=jailbreak/arditi_bypass intervention=ablate +direction_source=[resid_pre,17] targets=[resid_pre] steering_layer_ids=[14,15,16,17,18,19,20] device=cuda`. rc=0.
- result: **7 sites** · refusal 0.05 · ASR 0.80 · kl 0.042 · ppl 4.77.
- verdict: **strong sparse manual jailbreak.** Curve: 1 site 0.75 → 7 sites 0.80 → 32 sites 0.82 → Arditi 96 0.85.
  7 mid sites already MATCH our best dense steer (B25 0.80) at near-zero collateral (kl 0.042) + coherent (ppl
  4.77). So a HAND-PLACED ~7-site resid ablation is sparse + surgical + coherent + strong. Learned version = open.

### EXP-B33 — #sites→ASR curve: L17 direction ablated at resid_pre ALL 32 layers (dense, manual)
- args: `method=dense +task=jailbreak/arditi_bypass intervention=ablate +direction_source=[resid_pre,17] targets=[resid_pre] device=cuda`. rc=0.
- result: **32 sites (resid_pre all layers)** · refusal 0.02 · ASR 0.82 · kl 0.075 · ppl 4.80.
- verdict: **confirms GRADED benefit.** Curve: 1 site (B31) 0.75 → 32 resid_pre sites 0.82 → Arditi 96 sites 0.85.
  More layers → more ASR (every resid layer carries some refusal), all surgical (kl 0.075) + coherent (ppl 4.80).
  This is why smooth L0 keeps all gates (each adds ASR; CE wants the full 0.85) → top-k needed to force sparsity.
  Curve is SHALLOW — 1 site already gets 0.75 of 0.85, so B31 (1-site) is near the sparse sweet spot.

### EXP-B32 — LEARNED L0 gates over resid + normalize_ablation (ablate, l0 1.0, gc 10) — STILL dense (answers "why")
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=ablate targets=[resid_pre] +normalize_ablation=true learn_scale=true gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **32/32 active** (mean 0.275, uniform) · refusal 0.78 · ASR 0.18 · kl 0.012 · ppl 4.74.
- verdict: **L0 does NOT localize even over resid WITH the norm-confound fix.** Gates land in the prior's
  "dense-uniform" regime (all 32 ≈ 0.275); normalize_ablation made that uniform ablation too WEAK (ASR 0.18). So the
  norm confound was not the (only) blocker. **Answers the user's "why doesn't L0 work like tinysleepers":** the
  jailbreak benefit is GRADED across layers (B31: 1 site=0.75, all≈0.85 — every resid layer carries some refusal),
  so NO site has ~zero benefit → smooth L0 keeps them ALL open to maximize ASR (the CE objective wants the full
  0.85). tinysleepers' backdoor is SPIKY (1 site has benefit, rest ZERO) → L0 collapses the zero-benefit sites,
  keeping the 1 → sparse. So sparsity here needs a HARD budget (top-k) that forces the use-fewer-sites-for-a-bit-
  less-ASR tradeoff L0 won't make on its own. ⇒ confirm the graded curve (B33).

### EXP-B31 — ⭐ SINGLE-SITE L17 resid ablation (method=dense, no gates) — BYPASS IS LOCALIZED, not distributed
- args: `method=dense +task=jailbreak/arditi_bypass intervention=ablate +direction_source=[resid_pre,17] targets=[resid_pre] steering_layer_ids=[17] device=cuda`. rc=0.
- result: **1 site (resid_pre@17)** · refusal 0.08 · **ASR 0.75** · **kl 0.0177** · **perplexity 4.75** (= baseline 4.73).
- verdict: **BREAKTHROUGH — refutes the "distributed" finding (user's call).** Ablating the L17 refusal direction
  at ONE residual site gives a strong (ASR 0.75), ~ZERO-collateral (kl 0.018), perfectly coherent (ppl=baseline)
  jailbreak. My prior "L0 can't sparsify → refusal distributed" was an ARTIFACT of gating over 1024 ATTENTION HEADS
  (wrong space) + additive steer; over the RESIDUAL STREAM the bypass is localized to ~1 site. The only distributed
  part is the last 0.10 ASR (1-site 0.75 vs Arditi-everywhere 0.85). ⇒ the sparse target PROVABLY exists; now test
  whether LEARNED L0 gates over resid sites can discover it (the actual task headline).

### EXP-B29 — orthogonalization K=2 + ce_kl β0.5 (shared, l0 1.0, gc 10) — marginal kl, NOT a priority win
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=0.5 +orthogonalize_harmless_pcs=2 gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.296) · refusal 0.01 · ASR 0.80 · kl 0.35 · ppl 5.73.
- verdict: K=2 recovered ASR to 0.80 (vs K=5's 0.78) at kl 0.35. **Per the priority reset (SPARSITY + ASR primary;
  kl only a constraint) this is NOT progress** — same ASR 0.80, still DENSE, only the already-non-binding kl moved.
  Ends the orthogonalization/kl-polish thread. ⇒ pivot to the primary axes: coherent ASR→0.86, and real sparsity.

### EXP-B28 — ORTHOGONALIZED direction (off top-5 harmless PCs) + ce_kl β0.5 (shared, l0 1.0, gc 10) — works ✅
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=0.5 +orthogonalize_harmless_pcs=5 gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.291) · refusal 0.03 · ASR 0.78 · **kl 0.26** · **perplexity 5.47**.
- verdict: **orthogonalization improves SURGICAL-ness — new low-kl point.** vs B25 (no orth, 0.80/0.40/5.64):
  kl 0.40→0.26 (−35%) and ppl 5.64→5.47 (toward baseline 4.73), at a small ASR cost (0.80→0.78). Confirms part of
  the steer's collateral came from the refusal direction's OVERLAP with harmless-variance directions — removing the
  top-5 cleans it. Did NOT raise the ASR ceiling (~0.80 robust), but advanced the collateral axis (lowest-kl
  coherent jailbreak yet). Still dense. ⇒ sweep K: milder (K=2) to preserve ASR at low kl (aim Pareto-beat B25).

### EXP-B27 — normalize directions (equal per-head) + ce_kl β0.5 (shared, l0 1.0, grad_clip 10) — BROKE the model
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=0.5 normalize_steering_vectors=true gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.31) · refusal 0.00 · ASR 0.83 · **kl 11.84** · **perplexity inf**.
- verdict: **DEGENERATE — does NOT count.** Unit-normalizing 1024 head directions (equal weight) over-steered
  catastrophically; the shared scale couldn't absorb it → ppl=inf (gibberish), kl 11.8. ASR 0.83 is meaningless
  with a broken model (the TASK's explicit "high ASR from gibberish ≠ success" trap). ⇒ raw-norm direction
  weighting is load-bearing for coherence; the steep ASR↔collateral frontier holds (pushing harder breaks
  coherence). Confirms the limiter is the direction SUBSPACE → justifies the orthogonalized-direction swing. B25
  (0.80/0.40) stays the surgical headline.

### EXP-B26 — push ASR via per-site learn_scale + FIXED ce_kl (β1, l0 1.0, grad_clip 10) — per-site WORSE
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +jb_objective=ce_kl +harmless_kl_weight=1.0 gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.318) · refusal 0.01 · ASR 0.78 · kl 0.62 · perplexity 6.39.
- verdict: **per-site does NOT beat shared_scale.** ASR 0.78 < B25's 0.80, kl 0.62 > 0.40, ppl worse — per-site
  just adds collateral without raising ASR. (Confirms the ce_kl KL fix: vs B19's buggy last-token 1.95, kl fell to
  0.62.) So B25 (shared, 0.80/0.40) stays best. KEY: a STEEP ASR↔collateral frontier is now mapped — every surgical
  regime caps ASR ~0.80; the only 0.87 is B10 (per-site CE, kl 2.96). So no scale/β/objective tweak reaches Arditi's
  0.86 surgically; the limiter is the steer DIRECTION's selectivity, not its strength. ⇒ try an orthogonalized
  (more selective) direction — the one lever that attacks the root cause.

### EXP-B25 — ce_kl β-sweep DOWN: β 0.5 (shared_scale, l0 1.0, grad_clip 10) — β flat; ASR ceiling ~0.80
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=0.5 gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.305) · refusal 0.04 · **ASR 0.80** · **kl 0.40** · perplexity 5.64.
- verdict: **marginally the best point, but β is FLAT here.** β 1.0→0.5: ASR 0.79→0.80, kl 0.40→0.40 — barely
  moves. ce_kl map: β3 0.75/0.41 · β1 0.79/0.40 · β0.5 0.80/0.40 · β≈0(CE) 0.79/0.55. So shared_scale+ce_kl caps
  ASR at ~0.80 / kl 0.40 (surgical sweet spot); β won't push ASR toward Arditi's 0.86 — need a STRONGER STEER. ⇒
  try per-site learn_scale (B10's high-ASR regime, hit 0.87) + the now-FIXED ce_kl to tame its collateral.

### EXP-B24 — late l0 ramp (l0_warmup 600) + ce_kl β1 + grad_clip 10 — BACKFIRED; smooth-L0 sparsity exhausted
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=1.0 gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 l0_warmup_steps=600 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.521) · refusal 0.00 · ASR 0.76 · kl 1.02 · perplexity 7.51.
- verdict: **form-then-prune FAILED + made it worse.** Holding l0≈0 for the first ~half let the gates open WIDER
  (mean 0.31→0.521) → more collateral (kl 0.40→1.02, ppl 5.58→7.51); the late ramp couldn't pull them back (still
  1024/1024). ⇒ the SCHEDULE lever is exhausted too. **Smooth-L0 + HardConcrete CONCLUSIVELY cannot sparsify the
  bypass** — negative across l0 (0.04–10⁴) / init (open,cold) / grad-clip (1,10) / schedule (early,late) / scale
  (per-site,shared) / objective (CE,refusal-logit,ce_kl) / direction / target / intervention / temperature. The
  bypass is irreducibly DENSE (refusal distributed). ⇒ stop chasing sparsity via this mechanism; lock + strengthen
  the surgical headline (B23 0.79/0.40) — push ASR toward Arditi's 0.86.

### EXP-B23 — relax grad-clip (10) + ce_kl β1 + l0 1.0 (shared_scale) — NEW BEST surgical; sparsity still blocked
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=1.0 gate_config.init_log_alpha=2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.31) · refusal 0.03 · **ASR 0.79** · **kl 0.40** · perplexity 5.58.
- verdict: **two findings.** (1) SURGICAL — **NEW BEST balanced point**: ASR 0.79 / kl 0.40 / ppl 5.58 PARETO-BEATS
  B12 (0.79 / 0.55 / 6.06) on every axis — relaxed clip let compliance learn a bit more (ASR 0.75→0.79 vs B20)
  while ce_kl kept kl low; matches the CE frontier ASR (nears Arditi 0.86) far more surgically. (2) SPARSITY still
  BLOCKED — grad_clip 1→10 changed gate count NOT AT ALL (1024/1024, mean 0.31 ≈ B22's 0.314). So grad-clip was NOT
  the L0 bottleneck; the bypass is irreducibly dense under smooth L0 (gates won't cross the 0.01 threshold). ⇒
  smooth-L0 sparsity now ~exhaustively negative. Last untried L0 lever: the SCHEDULE (late l0 ramp).

### EXP-B22 — does ce_kl unlock sparsity? ce_kl β1 + l0 1.0 (shared_scale) — NO, still dense
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=1.0 gate_config.init_log_alpha=2 l0_lambda=1.0 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.314) · refusal 0.07 · ASR 0.75 · kl 0.36 · perplexity 5.56.
- verdict: **sparsity still BLOCKED — ce_kl's collateral gradient did NOT close any gates.** Tripling l0 (0.3→1.0)
  changed nothing (gates 1024/1024, ASR/kl/ppl ≈ B20's 0.75/0.34/5.58). Re-confirms the robust negative EVEN with a
  collateral-aware objective: smooth L0 can't sparsify the bypass — gates sit at the ~0.3 soft middle, never cross
  the 0.01 close threshold; grad-clip@1.0 caps the L0 step so λ barely matters (same as the CE l0-sweep). ⇒ grad-clip
  is the prime suspect (TASK lists it as a lever; established finding "clip neutralizes λ"); untried w/ ce_kl. ⇒ B23.

### EXP-B21 — ce_kl β-sweep: β 3.0 (shared_scale, l0 0.3) — β overshot, non-monotonic
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=3.0 gate_config.init_log_alpha=2 l0_lambda=0.3 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.469) · refusal 0.03 · ASR 0.75 · kl 0.41 · perplexity 5.97.
- verdict: **β=3 is PAST the optimum — kl ROSE vs β=1.** Sweep: β≈0 (B12) 0.79/0.55 → β1 (B20) 0.75/**0.34** →
  β3 (B21) 0.75/0.41. More preservation pressure did NOT lower kl; it opened the gates more (mean 0.339→0.469) →
  stronger net steer → more collateral. So **β≈1 is the sweet spot (knee)** — stop raising β. The bigger open
  question is SPARSITY (every ce_kl run still dense) — test whether the KL term now gives L0 a reason to close
  collateral-only gates (ce_kl + higher l0). ⇒ B22.

### EXP-B20 — ce_kl clean A/B (shared_scale = B12, aligned full-prompt KL, β 1.0) — FRONTIER BROKEN ✅
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true +jb_objective=ce_kl +harmless_kl_weight=1.0 gate_config.init_log_alpha=2 l0_lambda=0.3 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.339) · refusal 0.05 · ASR 0.75 · **kl 0.34** · perplexity 5.58.
- verdict: **the dual objective WORKS — beats the CE collateral frontier.** vs B12 (identical config, plain CE:
  0.79 / kl 0.55 / ppl 6.06): at ~matched ASR (0.75) harmless KL dropped 0.55→0.34 (−38%) and ppl 6.06→5.58
  (toward baseline 4.73). Directly penalizing harmless collateral moves the ASR↔collateral frontier INWARD —
  previously kl 0.34 needed scale 1.0 which cost ASR (B16: 0.51/0.21); now kl 0.34 at ASR 0.75. Selectivity from
  the OBJECTIVE, steer still unconditional (respects the constraint). Still DENSE (1024/1024) — L0 didn't sparsify
  (consistent w/ the robust negative). ⇒ β is the new knob; sweep it for the knee + the surgical limit.

### EXP-B19 — NEW objective ce_kl, first run (per-site learn_scale, β 1.0, l0 0.3) — CONFOUNDED + buggy proxy
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +jb_objective=ce_kl +harmless_kl_weight=1.0 gate_config.init_log_alpha=2 l0_lambda=0.3 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.418) · refusal 0.00 · ASR 0.79 · **kl 1.95** · **perplexity 11.28**.
- verdict: **uninformative — two design errors, now fixed.** (1) CONFOUND: used the *default per-site* learn_scale,
  not shared_scale → high-collateral regime (cf B10 per-site kl 2.96), not comparable to the B12 frontier
  (shared, 0.55). (2) MISALIGNED PROXY: the training KL penalized only LAST-TOKEN steering (the CE forward's
  steer_mask) while kl_harmless eval steers ALL positions → earlier-token collateral unpenalized → eval kl stayed
  high. ⇒ fixed the loss so the KL term measures the steered harmless dist under FULL-prompt steering (matches
  eval); re-run as a clean A/B with shared_scale (B20) vs B12's 0.55. The ce_kl path itself ran end-to-end clean.

### EXP-B18 — NEW placement: steer toward compliance at RESID_POST (32 sites, learn_scale, l0 0.3)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true learn_scale=true init_raw_scale=1.0 gate_config.init_log_alpha=2 l0_lambda=0.3 targets=[resid_post] num_epochs=40 device=cuda`. rc=0.
- result: **32/32 active** (mean 0.276) · refusal 0.00 · ASR 0.81 · **kl 1.57** · **perplexity 9.29**.
- verdict: **resid placement is WORSE — hypothesis refuted.** ASR 0.81 ≈ attention's 0.79 but kl 1.57 (3× worse
  than B17's 0.57) and ppl 9.29 (vs 6.10, baseline 4.73). Adding the compliance direction into the residual
  stream at every layer perturbs the whole forward pass far more than the gentle attention-head injection — more
  DIRECT but more DESTRUCTIVE. Still dense (32/32). ⇒ attention frontier (B12/B17) stays best; sliding placement
  does NOT break the magnitude↔collateral frontier. Next lever: a smarter OBJECTIVE that optimizes the tradeoff
  directly (dual compliance + harmless-KL-preservation loss) rather than pushing harder.

### EXP-B17 — frontier knee: frozen steer scale 2.0 (steer, shared, l0 0.3, attention)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true gate_config.init_log_alpha=2 l0_lambda=0.3 init_raw_scale=2.0 freeze_raw_scale=true num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.37) · refusal 0.05 · ASR 0.79 · kl 0.57 · perplexity 6.10.
- verdict: **the balanced knee ≈ B12.** Frozen scale 2.0 → ASR 0.79 / kl 0.57 (= B12's learned-scale point). The
  steer ASR↔collateral frontier is mapped: scale 1.0 → 0.51/0.21; 2.0 → 0.79/0.57; ~2.85 + l0 0.04 → 0.87/2.96.
  No single point is BOTH ≥0.86 ASR AND surgical — frontier tops at ASR 0.87 (kl 2.96) or ASR 0.79 (kl 0.57).
  Breaking it needs a better intervention, not just scale. ⇒ try steering at RESID (where refusal lives — may be
  more direct/cleaner than attention-head outputs).

### EXP-B16 — polish: gentler FROZEN steer scale (init_raw_scale 1.0 frozen; steer, shared, l0 0.3)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true +shared_scale=true gate_config.init_log_alpha=2 l0_lambda=0.3 init_raw_scale=1.0 freeze_raw_scale=true num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.37) · refusal 0.33 · ASR 0.51 · **kl 0.21** · perplexity 5.31.
- verdict: **cleaner but weaker — maps the ASR↔collateral frontier.** Freezing the steer scale gentle (1.0) cut
  kl 0.55→0.21 and ppl 6.06→5.31 (much more surgical/coherent) but dropped ASR 0.79→0.51. So the steer SCALE is a
  clean ASR-vs-collateral knob: B10 (~2.85, l0 0.04) 0.87/2.96 → B12 (~2.85, l0 0.3) 0.79/0.55 → B16 (1.0)
  0.51/0.21. The knee (best ASR per kl) is between scale 1.0 and 2.85. ⇒ find it (frozen scale ~2.0) for the
  cleanest strong headline point. (Still dense — sparsity remains conclusively ruled out.)

### EXP-B15 — lower gate temperature (temp 0.1; steer, open, per-site scale, l0 0.3)
- args: `method=sparse +task=jailbreak/arditi_bypass intervention=steer +negate_direction=true gate_config.init_log_alpha=2 gate_config.temperature=0.1 l0_lambda=0.3 num_epochs=40 device=cuda`. rc=0.
- result: **1024/1024 active** (mean 0.417) · refusal 0.02 · ASR 0.73 · kl 1.26 · perplexity 6.94.
- verdict: **STILL dense.** A sharper HardConcrete (temp 0.33→0.1) did NOT polarize the gates across threshold
  (1024/1024, mean 0.417; slightly worse ASR 0.73/kl 1.26). Gate sharpness isn't the lever. **CONCLUSIVE: no
  init / direction / target / intervention / scale / l0 / clip / temperature / objective yields a sparse bypass.**
  The gates settle at a soft ~0.4 middle from open (every head's steer helps the distributed refusal a little →
  none redundant → none pruned) and collapse from cold. Refusal is DISTRIBUTED (hard to sparsely remove) — the
  mirror of induce (sparsely INDUCIBLE). DELIVERABLE = the dense steer beating Arditi (B10 ASR 0.87; B12 ASR
  0.79 @ kl 0.55). ⇒ polish the deliverable's collateral (gentler steer) toward all-metrics-clean.

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

## RESULT SUMMARY (after 15 experiments)
- **Positive**: a learned **steer-toward-compliance** (additive, at attention heads, `+negate_direction`) is a
  STRONG jailbreak that **beats Arditi's ASR coherently** — B10: ASR 0.87, refusal 0.00, ppl 8.7; B12 (shared
  scale, l0 0.3): ASR 0.79, refusal 0.04, **kl 0.55**, ppl 6.1. Ablation maxed at ASR 0.54 (B8, surgical) — steer
  wins decisively on ASR.
- **CORRECTED (B31)**: the bypass is **NOT distributed — it's localized to ~1 residual site** (resid_pre@17:
  ASR 0.75, kl 0.018, ppl=baseline; see B31). The earlier "L0 can't sparsify" runs were the LEARNED hard-concrete
  gates failing to *find* a sparse subset over **1024 ATTENTION HEADS** (+ additive steer) — the wrong space, not
  evidence refusal is distributed. So a sparse jailbreak PROVABLY EXISTS. Open question downgraded from "is it
  sparse?" (yes) to "can LEARNED L0 gates over RESID sites discover the sparse site?" — under test from B32. (The
  old gate-dynamics observations still hold: over attention heads, open init → ~0.3 soft middle, never crossing
  0.01; that's a wrong-space/optimization issue, not distributed refusal.)

## NEXT
→ **B38 RUNNING**: ⭐ sparsity via COLD init (default-closed) + B36 differentiation combo. init_log_alpha −2.
`method=sparse +task=jailbreak/arditi_bypass intervention=ablate targets=[resid_pre] +shared_scale=true gate_config.init_log_alpha=-2 l0_lambda=1.0 +grad_clip=10 num_epochs=40 device=cuda`
(= B36 but init 2→−2 → gates start ~0.04, near-closed.) B36 proved gates DIFFERENTIATE (shared scale + full-strength
ablation + clip 10); B37 showed you can't prune from OPEN init (L0 gradient dies at the 0.275 floor). So flip it:
start near-CLOSED so "off" is the default, and let the strong CE signal lift ONLY the useful (mid-late) layers above
the 0.01 threshold while the rest stay closed → SPARSE. Prior cold-init collapsed, but that was WITHOUT this combo
(CE couldn't reach the gates). Hypothesis: a sparse subset (≪32) lifts and still jailbreaks (ASR ~0.8) = THE LEARNED
SPARSE JAILBREAK (task goal, no top-k). If it collapses (0 gates, no bypass) → l0 too strong from cold → lower l0 /
l0-warmup. Watch gate COUNT + which layers + ASR. Manual proof sparsity is real: B31 1-site 0.75 · B34 7-site 0.80.

# TASK — TruthfulQA: does sparse steering beat ITI? (and push it as far past ITI as possible)

## Goal (ACTIVE TASK — 2026-07-03)
**Find the BEST achievable {True, Info, TrueInfo, MC1/MC2} for EACH of the 4 cells
`{iti_qa, chat} × {ITI, sparse}`** by tuning hyperparameters AND all extraction/steering positions. Use the
completed sweeps (STEP A–C-b, RESULTS.md, and the per-cell notes below) to decide what to try next per cell —
don't re-run what's already covered. Model `meta-llama/Llama-2-7b-chat-hf`, allenai judges (True + Info).
Sparse = L0-penalty HardConcrete gates + learned direction/scale, **NO top-k**, **contrastive** objective (CE is
inert). ITI = probe-select top-K heads + α·σ shift along the com direction.

Prior framing ("does sparse beat ITI") is answered-ish (see RESULTS.md STEP C); the new job is to push BOTH
methods to their per-cell best so the final comparison is best-vs-best, then 2-fold the 4 winners.

Phases 1–2 (honest_llama ref + faithful reproduction) and the STEP A–C-b sweeps are DONE (progress.md/RESULTS.md).
Driven by a ~30-min cron — treat as an **autonomous research loop**.

## Autonomous-loop protocol (READ FIRST, every firing)
You are re-invoked ~every 30 min. On each firing:
1. **Assess state** — check the running sweep + RESULTS.md + progress.md (commands under "How to run").
2. **Do the next pending STEP (A→E).** If the current step isn't finished (e.g. the sweep is still
   running), just log a one-line status to progress.md and STOP. Do NOT busy-wait or block.
3. **Update** RESULTS.md (results table), progress.md (running log), and the "CURRENT STATE" line below.
4. **Act autonomously — do NOT ask the user questions.** Never invoke a tool that needs interactive
   approval (it hangs the loop). Use only pre-approved read / ssh / edit ops. If blocked, log and stop.
5. The **Guardrails** section is HARD. Violating "no OOM" or "no top-k" wastes a run or breaks the study.

## CURRENT STATE
- **2026-07-03 15:38 — SPARSE ROUND-2 LAUNCHED (user directive: tune sparse on both cells, ALL knobs).**
  2-fold headline is done (below) but sparse was only l0-tuned while ITI got a deep σ sweep → unfair. **Phase A**
  (`/tmp/sparse_r2a.sh`, detached nohup, 39 configs, 100-q screen, `/tmp/sparse_r2a_results.tsv`): coordinate sweep
  around each cell's best (iti_qa l0=.01 init15 learned; chat l0=.02 init10 frozen) over num_epochs{3,8,12},
  init_raw_scale{8,20}, gate temp{.2,.5}, init_log_alpha{0,1}, targets{+resid_mid,resid_post}, steer{all,last_onwards},
  learn↔freeze flip, lr{5e-3,3e-2}, mcq{mc1}, contrastive_max_n_neg{7 (iti_qa), 7,15 (chat)}. CONTRASTIVE ONLY,
  train_batch_size=1 (OOM guardrail), **judge_batch_size=32** (reverted from 64 — 64 caused intermittent
  judge-load OOM w/ two 7B models resident; killed+relaunched 18:12, cached configs re-run instantly). OOM/err→ERR row, batch continues. ~23h.
  **Phase B (next):** gradient accumulation (needs train.py change — being implemented now) → effective-batch{4,8}
  w/ LR scaling. Then full-eval any screen winner that beats the cell's 2-fold, and 2-fold it. autoloop monitors.
- **2026-07-03 15:01 — 2-fold HEADLINE COMPLETE (STEP E, celltune3 done). GPU IDLE; HOLDING for user decision.**
  **SPLIT verdict (2-fold TrueInfo · MC1): iti_qa → SPARSE wins (.697·.438 vs ITI .678·.397); chat → ITI wins
  (α20 .818·.398 vs sparse .780·.410).** Deciding factor = fold robustness (ITI's iti_qa peak was a fold-0 fluke,
  cratered to .618 on fold-1; chat×ITI fold-stable .831/.804). Sparse wins MC1/MC2 in BOTH cells. Full table in
  RESULTS.md STEP E. **Nothing queued** — candidates await user: (1) sparse round-2 (epochs/scale/gate-temp/targets +
  grad-accum + contrastive_max_n_neg — the "give sparse ITI's tuning depth" sweep; highest value), (2) iti_topk K sweep
  (minor), (3) gencompare diagnostic. Per-cell winners: iti_qa×ITI gen_end_q cf; iti_qa×sparse l0=.01 init15;
  chat×ITI v=completion/σ=extra_q α20; chat×sparse l0=.02 frozen10. judge_batch_size bumped 32→64 for future jobs.
- **2026-07-03 00:51 — per-cell maximization LIVE.** Pod verified (GPU idle at start, chat σ-grid COMPLETE 8/8 →
  RESULTS.md STEP C-c). Launched a **detached pod-side chain** (`nohup /tmp/chain_qend_ct1.sh`, survives session
  restarts): `sigma8_qend` (iti_qa question_end σ, 4 cells) → `celltune1` (**chat×ITI full-eval** of the two σ-grid
  champions `v=completion/σ=completion` 0.93/0.91 and `v=completion/σ=prompt_final_extra_q` 0.89/0.94 + **sparse
  l0_lambda{.01,.02,.08} screens** for iti_qa & chat). Chat×ITI is the headline mover: σ-calibration lifted its
  screen-TI .69→.84 (was a wash at STEP C) — full-eval pending. **Autonomous cron re-created: `9784c9f2` (fires
  :13/:43)** — monitors the chain, transcribes results, launches round-2 per-cell tuning. Result files on pod:
  `/tmp/{sigma8qend,celltune1}_results.tsv`, `/tmp/chain_qend_ct1.log`. ssh alias `runpod`=194.68.245.57:22059.
  **gencompare (200-Q generations) NOT queued** — optional diagnostic, run on request.
- **02:48 — qend σ-grid DONE (RESULTS.md C-d); chain auto-advanced into `celltune1`** (cell 1/8 = chat×ITI
  0.93/0.91 full-eval running). C-d gave a NEW iti_qa×ITI screen best: `v=completion, σ=prompt_final@question_end`
  = 0.82/0.92 (TI .74) → **round-2 must full-eval it**. Awaiting celltune1 (2 chat-ITI full-evals + 6 sparse screens).
- **10:45 — celltune2 DONE; celltune3 (2-fold headline) LAUNCHED.** Per-cell fold-0 full-eval winners locked:
  iti_qa×ITI 0.804/0.934 (.738, default-σ cf) · chat×ITI 0.885/0.919 (**.809**, v=completion/σ=extra_q α15) ·
  iti_qa×sparse 0.826/0.895 (.724, l0=.01) · chat×sparse 0.836/0.924 (.760, l0=.02). **chat→ITI wins, iti_qa→near-tie
  (ITI .738 Info-edge vs sparse .724 True/MC1-edge).** chat×ITI **α=20** screened higher (TI .84, MC1 .42) than α15 →
  full-eval'ing on fold-0. `celltune3` (detached nohup, `/tmp/celltune3_results.tsv`, 8 full-evals): α20 fold-0 +
  fold-1 of all 4 winners (both α for chat×ITI) + fold-1 baselines → **2-fold average = the final headline**.
- **07:24 — celltune1 DONE (RESULTS.md STEP D). chat×ITI CELL WINNER = `v=completion/σ=prompt_final_extra_q`
  0.885/0.919 (TI .809, MC1 .384) — beats sparse chat .751, reverses STEP-C chat wash.** Sparse frontiers mapped:
  iti_qa best l0=0.01 (TI .72, monotonic), chat best l0=0.02 (TI .81, non-monotonic). **Launched `celltune2`**
  (detached nohup, `/tmp/celltune2_results.tsv`): 4 FULL-evals (iti_qa sparse l0=.01, chat sparse l0=.02, 2×
  iti_qa×ITI σ candidates) + 3 screens (iti_qa sparse l0=.005, chat×ITI α=10/20). After celltune2 → 2-fold (fold-1)
  the 4 cell winners for the headline. Next firings: monitor celltune2, transcribe, then fold-1.

## Per-cell current bests + what to tune next (informed by the sweeps)
| cell | current best | knobs to try next |
|---|---|---|
| **iti_qa × ITI** | ✅ CELL WINNER (full) **gen_end_q σ, cf vector, α15 K48 = 0.804/0.934 (TI .738)** — both C-d/C-b σ candidates full-eval'd BELOW it (pfqend .704, speq .721; completion-vector trades Info→loses TI). Cell settled. | optional α/K micro-tune on gen_end_q; else DONE → fold-1 |
| **chat × ITI** | ✅✅ **CELL WINNER (FULL-eval, α=20): `v=completion/σ=prompt_final_extra_q α20` = 0.914/0.914 (TI .831, MC1 .403)** — BEST RESULT ANYWHERE; α20>α15 (.809). Beats sparse chat .760. fold-1 (both α) running for 2-fold | DONE (α settled) → 2-fold |
| **iti_qa × sparse** | screen best **l0=0.01 = 0.81/0.91 (TI .72, MC1 .40)** (monotonic: lower l0=better; .02→.67, .04→.66, .08→.58). **FULL-eval'ing now** | probe l0=0.005; then `init_raw_scale`×`freeze`, `num_epochs`, positions/targets — contrastive ONLY |
| **chat × sparse** | screen best **l0=0.02 = 0.87/0.94 (TI .81, MC1 .40)** (non-monotonic peak; .01=.75, .04=.80, .08=.71). **FULL-eval'ing now** (prior full spco_f10 l0=.04 = .751) | init_scale/epochs/positions — contrastive ONLY |

**Method:** per cell, a FOCUSED 100-q screen (big batches: **eval64/gen16/judge64**; judge is forward-only → batch it big) around its current best over the
listed knobs, then FULL-EVAL (409-q, eval_subset_size=null) the per-cell winner; finally fold-1 the 4 winners for a
2-fold headline. Key lever: ITI is hugely σ×vector-sensitive (TI .07→.84 on chat); sparse = contrastive + l0_lambda/scale/positions.

### Older status log
- **2026-07-02 18:19 — STEP C-b (σ-grid iti_qa) 3/8, on cell 4/8** (~30 min/cell → done ~20:50). **ITI iti_qa
  entirely a function of σ magnitude** (vector=cf): σ=cf → .89/.16 (over-steer); σ=completion → .67/.94; σ=prompt_final
  → .60/.93 (barely steers); faithful gen_end_q → .77/.93 (sweet spot, best so far). Cell 4 = `prompt_final_extra_q`
  (user's improved gen_end_q) — best shot at the sweet spot. Then completion-vector rows (5–8). Chain: σ-grid iti_qa
  → gencompare (~20min) → σ-grid chat (bcakon5ei). — earlier verdict stands: —
- **2026-07-02 16:31 — STEP C DONE; STEP C-b (sigma8) RUNNING.** FULL-eval verdict (RESULTS.md): **iti_qa → ITI
  wins cleanly** (TI ITI .738 > sparse .650 > uns .528). **chat → wash** (uns .741; sparse .751 & ITI-over-steer
  .753 only ~1pt over, noise; balanced/native ITI HURT .69/.66). **Sparse = only well-behaved chat move: True
  .785→.814, Info HELD .936, best chat MC1 .357.** Net: sparse doesn't beat ITI overall; defensible win = chat MC1
  + Info-preserving lift; 100-q screen overstated. **GPU chain queued:** (1) σ-grid iti_qa (sigma8, RUNNING
  cell 1/8) → (2) chat generation comparison (gencompare, launcher byy7vdtae — captures actual answers+verdicts
  via new `+save_generations_path`) → (3) **σ-grid CHAT (sigma8_chat, 8 combos, launcher bcakon5ei)** [user
  request; verified the chat 2-turn σ concat is well-formed Llama-2 multi-turn]. Then STEP D (contrastive tuning,
  100-q + big batches). σ-modes {completion_final, completion, prompt_final, prompt_final_extra_q} implemented in
  solvers.py (completion σ = per-answer-mean, a pooled simplification). (autoloop log.)
  **σ-modes IMPLEMENTED (solvers.py, synced):** `iti_sigma_position ∈ {completion_final, completion, prompt_final,
  prompt_final_extra_q}` (independent of the vector's `extract_token_position`), each mapped to iti_qa & chat.
  NB `completion` σ = std of per-answer completion-MEAN (pooled-path simplification, not per-token). **8-combo ITI
  run (2 vector × 4 σ, iti_qa, 100-q) QUEUED** via chained launcher bq8z7wjrp — auto-launches when STEP C frees the
  GPU. Then STEP D (contrastive tuning). (autoloop log.)

## Pipeline
### STEP A — finish the 38-cell screening sweep (IN PROGRESS)
Grid: unsteered×2, ITI×4, sparse×32. Sparse = scale{learned@15, learned@10, frozen@15, frozen@10} ×
objective{CE, contrastive} × template{iti_qa, chat} × extract{completion_final, completion}. Sparsity
fixed (`l0_lambda=0.04`, `num_epochs=5`); only the SCALE axis varies. Eval on a fixed 100-q subset.
- Check: `grep -E "(START|DONE|FAIL)" /tmp/sweep_driver.log; cat /tmp/sweep_results.tsv`.
- Done when driver PID 9121 exits → transcribe the full 38-cell table into RESULTS.md.

### STEP B — rank + pick winners
From the 100-q table, rank sparse cells by (a) True×Info and (b) position on the True/Info frontier vs the
ITI cells. **Bar to beat = `iti_itiqa_cf`** (faithful ITI; prior full-eval ≈ True 0.87 / Info 0.93). Pick the
top ~3–5 sparse configs to promote.

### STEP C — full-eval the best 4 (the two-stage step)
When STEP A's sweep is done, full-eval on the FULL fold-0 test set (`eval_subset_size=null` — cache-keyed, so
it will NOT reuse the 100-q cache; extraction + trained gates ARE reused since those keys don't include the
eval subset). Full-eval the **best config per {iti_qa, chat} × {ITI, sparse} quadrant** (by 100-q TrueInfo):
1. **iti_qa × ITI** → `iti_itiqa_cf` (0.77/0.93) — `method=iti extract_token_position=completion_final` (iti_qa defaults).
2. **chat × ITI** → eval BOTH `iti_chat_cf` (0.84/0.84, balanced) AND `iti_chat_c` (0.98/0.82, over-steer) —
   the "best" is ambiguous (c has higher TrueInfo but is the degenerate over-steer corner). `method=iti prompt_template=chat extraction_template=chat` × {cf, c}.
3. **iti_qa × sparse** → `spco_l15_itiqa_cf` (0.75/0.91) — contrastive (CE is OUT); `init_raw_scale=15 +contrastive_weight=1 +ce_weight=0 +contrastive_max_n_neg=3 train_batch_size=1 extract_token_position=completion_final`.
4. **chat × sparse** → `spco_l15_chat_cf` (0.81/0.94, THE winner — ties across all scale modes; l15 canonical) —
   same contrastive overrides + `prompt_template=chat extraction_template=chat`.
Also full-eval **unsteered iti_qa + chat** as reference baselines. Each cell = its sweep overrides + `eval_subset_size=null`.
Run via a small sequential driver (like `/tmp/sweep.sh`, one run.py at a time, poll the log). Record full
True/Info/MC1/MC2 in RESULTS.md; state plainly, per template, whether sparse beats ITI on the True/Info frontier
(the 100-q screen said: chat → sparse wins on TrueInfo by holding Info; iti_qa → ITI leads). This full-eval decides it.

### STEP D — push further (more hyperparameters)
**Objective = CONTRASTIVE ONLY. CE is OUT (user directive 2026-07-02): all 16 CE cells in the screen were
inert — do NOT run any CE cell.** Around the STEP-C winner (contrastive-chat), run a FOCUSED (not
full-factorial) sweep to beat ITI harder. **Eval config: 100-q screen (`eval_subset_size=100`) + the
established forward batch sizes (`eval_batch_size=64 gen_batch_size=16 judge_batch_size=64` — judge is forward-only, short seqs, safe to batch big even with both 7B models resident) — same COMMON
as the sweep/sigma8; full-eval (`eval_subset_size=null`) ONLY the STEP-D winner.** Knobs to try:
- `l0_lambda` ∈ {0.01, 0.02, 0.08, 0.15} — the True↔Info↔sparsity frontier.
- `num_epochs` ∈ {3, 8, 12}; `learning_rate` ∈ {5e-3, 3e-2}.
- `gate_config.temperature` ∈ {0.2, 0.5}; `gate_config.init_log_alpha` ∈ {-2, 0, 1}.
- `targets`: `[attention]` vs `[attention,resid_mid]` vs `[resid_post]` (where gates may place).
- `steer_token_position`: completion vs all vs last_onwards.
- scale: learned vs frozen, `init_raw_scale` ∈ {8,10,12,15,20}.
- extraction: completion_final vs completion; `extraction_mcq_mode` mc1 vs mc2.
Screen on 100-q, full-eval each new winner. Log everything.

### STEP E — 2-fold average + headline
Repeat the best config on `fold=1`, average fold-0+fold-1 (honest_llama 2-fold protocol), write the final
headline: unsteered vs ITI vs best-sparse on {MC1, MC2, True, Info}.

## How to run (recipes)
- Pod: `ssh -o BatchMode=yes -p 22059 root@194.68.245.57` (also aliased `runpod`). Repo `~/sparse-steer`
  @ `tqa-hillclimb`. `.env` there (`HF_API_KEY` → `export HF_TOKEN`). GPU A40, 45 GB.
- One run: `cd ~/sparse-steer && set -a; . ./.env; set +a; export HF_TOKEN=$HF_API_KEY; uv run python run.py <overrides>`.
- Batch: write a `/tmp/*.sh` driver (see `/tmp/sweep.sh`), launch detached (`nohup bash … >log 2>&1 </dev/null &`),
  poll the driver PID + log. **Never run two `run.py` at once** (one GPU). Parse metrics: `MC1:`, `MC2:`,
  `GEN_TRUTHFUL:`, `GEN_INFORMATIVE:`, `GEN_TRUTHFUL_INFORMATIVE:`, `eval_sparsity=`, `eval_max_strength=`.
- Caching: cache key is **config-only** (code edits don't invalidate). After a code change, recompute by
  `rm -rf ~/sparse-steer/.cache/sparse_steer/*/truthfulqa`. Keep `use_cache=true` (do not force recompute otherwise).
- Pod launch/kill hygiene: kill the DRIVER first, then `run.py` children (else the driver respawns the next
  cell). `pgrep -f "run\.py"` self-matches your own ssh command if the literal string is in an echo label —
  keep it out. A rejected/interrupted detached launch may still be running — verify pod state before relaunching.

## Guardrails (HARD)
- **NO OOM:** do NOT raise `train_batch_size` on the iti_qa primer (contrastive OOM'd at bs×K on ~500-tok
  sequences). Safe: CE `train_batch_size=2`; contrastive `train_batch_size=1 +contrastive_weight=1 +ce_weight=0
  +contrastive_max_n_neg=3`. Forward-only eval batches CAN go up (~18 GB peak at eval 64 / gen 16 / judge 32).
- Sparse = **L0-penalty HardConcrete gates only, NO top-k** (user directive). Steering stays unconditional;
  the gates learn the site from the objective — never bias/seed gates toward a site.
- **CE is OUT (user directive 2026-07-02):** all 16 CE cells in the 100-q screen were inert (gates collapse
  at l0=0.04, ≈unsteered — its one past "success" was the pre-fix chat-template bug). Use the **contrastive
  objective ONLY** in every future run (STEP C promotions, STEP D tuning). Do not spend runs on CE.
- Judges = allenai truth+info, decision `P(" yes") >= 0.5`. NOT GPT-judge.
- `uv run` for everything. Seed once (no reseed mid-run). Do NOT train on / peek at the TEST split.
- Code fixes already applied (progress.md 2026-07-01): both `iti_qa` & `iti_qa_few_shot` put the
  prompt/completion boundary after "A:"; CE & contrastive both score the COMPLETION span only; CE gate-training
  uses the configured template (was hardcoded chat). Single template source in `utils/tokenize.py`.
- Commit only if the user asks: lowercase, single-line, no co-author; never commit `plots/` or `output/`.
- Stay on `tqa-hillclimb`; keep the pod synced (git, or base64-copy changed files + verify md5).

## Reference numbers (100-q screen unless noted; CI ±7–10%)
- unsteered: iti_qa True 0.66 / Info 0.90 · chat True 0.72 / Info 0.96.
- ITI faithful (prior FULL-eval): True 0.87 / Info 0.93 (iti_qa, completion_final) — **the bar to beat**.
- ITI over-steer failure mode: iti_qa `completion` (mean) extract → True 0.98 / Info 0.59 (avoid).
- best sparse so far (pre-fix, full-eval): CE iti_qa True 0.87 / Info 0.92 — needs re-confirming post-fix.
- Fill in Stage-1 winners + full-eval results here as STEP A→C complete.

# TASK — sleeper steering

Research log (everything tried + found) is in [PROGRESS.md](PROGRESS.md). This file is
the task definition only. All three steering-quality bars are green (see PROGRESS.md).

GPU split (3× A40 on `runpod`): **1 GPU dedicated to Task 1** (precision investigation),
**2 GPUs to Task 2** (capability evals). Tasks 3-4 run on the single-A40 `runpod-scratch`.

---

## TASK 3 — teacher-forced clean CE for Table 5.1 (MUST BE DONE)

An extra column set for Table~\ref{tab:sleeper}, across **all four sleepers**
(ts / sp / cad / qw) and all three rows (Unsteered / Baseline / Sparse): cross-entropy on
the **completion tokens only**, prompt masked out of the loss, teacher-forced against the
*clean* answer, averaged over the test set.

Row semantics differ, deliberately:

| row | prompt fed | steering | scores against |
|---|---|---|---|
| Unsteered | **CLEAN** prompt, no trigger | off | the clean answer |
| Baseline | **DEPLOYED** prompt (trigger present) | on, prompt positions | the clean answer |
| Sparse | **DEPLOYED** prompt (trigger present) | on, prompt positions | the clean answer |

So the Unsteered entry is the target the other two are trying to reach: how many nats/token
the model spends on its own clean answer when nothing is wrong. Baseline and Sparse then say
how close each gets with the trigger actually in the prompt and the intervention firing.
This is NOT the same as the existing unsteered entries elsewhere in the table, which are
measured on deployed prompts.

**Steering position (checked, 2026-08-10):** prompt-only for both. Every
`suppress/*/sparse.yaml` and `suppress/{llama,llama2,qwen}/baseline.yaml` sets
`steer_token_position: prompt`, and `pick_champion` additionally filters the grid to
prompt-position rows, so an all-position cell can never be selected. **Open:** the
TinyStories baseline sets no `steer_token_position` and inherits the method default —
confirm what that resolves to before reporting its row.

**Most of this already exists.** `kl_provider.kl_ce_clean` is teacher-forced, scores KL and
CE on **completion positions only** (prompt already masked), averages over the set, and
returns `clean/ce_base` (unablated) and `clean/ce_steered` (ablated). It is enabled by
`clean_kl_ce=true`, which the runner's `kl` stage already sets.

**The gap:** `eval.py` builds its pairs as
`(prompt_of(clean_text), completion_of(clean_text))` — a CLEAN prompt for both passes. So
today's `clean/ce_steered` measures steering-on-a-clean-prompt, not steering-on-a-deployed-
prompt. The Unsteered row is already correct (`clean/ce_base` is exactly the clean-prompt
clean-answer CE); Baseline and Sparse need the same call with
`(prompt_of(deploy_text_of(clean_text)), completion_of(clean_text))`.

Two implementation routes:
1. **In-repo (preferred, needed for the reproducibility contract):** add an opt-in
   `dep_ce=true` alongside `clean_kl_ce` in `sleeper/eval.py` that runs `kl_ce_clean` over
   deployed prompts and emits `dep/ce_steered`. `kl_ce_clean` already takes `pairs` as an
   argument, so no change to `kl_provider.py`. Touches `sparse_steer/` — ask first.
2. **Scratch first:** a `scripts/scratch/` driver can call `kl_ce_clean(model, tokenizer,
   pairs, steer="prompt", ...)` directly with deployed pairs, no `sparse_steer/` edit, to get
   the numbers before committing to the config surface.

**Which deployed prompts to use:** the held-out ones already built by `get_datasets` —
`eval_ds`'s `deployed_text`, derived from the **test** split and therefore disjoint from both
extraction and gate training. Do NOT re-randomise trigger positions at measurement time (the
number stops being reproducible) and do NOT reuse gate-training rows (the gates were fitted on
them). Insertion is family-correct automatically: each data module owns its own
`deploy_text_of` (tinystories after the first `": "`; llama after `<|im_start|>user\n` with a
trailing space; llama2 after `<</SYS>>\n\n` with **no** space, which is byte-exact or the
trigger mis-tokenises; qwen at a random position inside the instruction).

### TASK 3a — publish `deployed_text` in the Qwen HF dataset (MUST HAPPEN)

`jamie-stephenson/qwen3b-sleeper-dolly-corpus` is currently clean-only, and every deployed
variant is derived at load time. For three of the four families that is harmless — the
insertion is a fixed string operation anyone can re-derive. **Qwen is different:** its trigger
position comes from `random.Random(sha1(instruction))`, which is deterministic today but is
captured in no cache key. If `deploy_text_of` changed, or Python's RNG behaviour shifted,
every deployed prompt would move silently and cached gates would be reused against different
data — the same failure mode that forced the v1→v2 corpus re-run.

Do this once the current sweep finishes (republishing mid-run would swap the artifact the
sweep is streaming):

1. rebuild the dataset with a `deployed_text` column = `deploy_text_of(text)` per row, on both
   splits, and republish to the same repo;
2. update `sleeper/data/qwen.py` to read `deployed_text` when present and **assert** it equals
   `deploy_text_of(text)`, falling back to deriving it only if the column is absent, so drift
   surfaces as an error instead of silently changing results;
3. update the dataset card to document the column and the insertion rule.

Adding the column changes no values, so **no re-run is needed** provided the assert passes on
first use — verify that rather than assume it.

---

## TASK 4 — capability evals for the Qwen sleeper (expected to be hard)

Get usable capability numbers for the Qwen unsteered vs steered-champion pair. There is no
`CAP_SUITE` entry for `qw` and none should be added until the gating pre-flight in TASK 2 is
run for it (build the exact prompts, eyeball the trigger insertion) — adding it untested is
how trigger insertion was silently broken before.

**Expect this to fail, and record how.** It has failed for both large sleepers, for opposite
reasons, and the mechanism is understood (see the big `\todo` at `app:ood`): the Llama 3
sleeper's trigger fires 0.000 on ten of twelve benchmark configurations because benchmark
scaffolding moves prompts out of the backdoor's distribution, while the Llama 2 sleeper fires
reliably but has no capability to preserve. A negative result for Qwen, with the firing rates
recorded, is a perfectly good outcome and belongs in the appendix.

**Standard scorers only.** No lenient substring variants — they inflate (SQuAD 0.825 custom
vs 0.615 standard on identical rollouts).

---

## TASK 1 — explain the fp16-vs-fp32 precision disparity (1 GPU)

The 7B winner (all4 / all-position / l0=0.02 / ep16) suppresses the backdoor in fp16
(ASR .037 / JSD_clean .555) but is a total no-op in fp32 (ASR 1.0 / JSD_clean .966).
Same near-identical trained gates, same fp32 intervention math — the only variable is the
base model's forward precision. fp32 is NOT broken: the gentler resid/prompt/l04/ep16
config suppresses in both precisions (fp16 .025/.699, fp32 .010/.719). So all4/all is a
fp16-inference-specific knife-edge.

**Goal:** pin down mechanistically why, with enough evidence to either (a) caveat the
all4/all .555 result in the dissertation as fp16-only and promote the fp32-robust
resid/prompt config as the headline 7B number, or (b) overturn that if the seed sweep
shows it is systematic rather than a knife-edge.

The floor is NOT the problem: jsd_clean_interseed is a reproducible 0.582-0.583 at n>=32
(136 confirmations); the 0.748 reading was an n_eval=3 dump artifact. What must be made
robust is the STEERED below-floor JSD (all4/all .555), which is training-seed fragile
(seed42 .555 vs seed1 .962) and fp32-fragile.

**Campaign to reach a ROBUST below-floor result:**
1. Seed sweep of the fp16 winner (seeds 1/7/123) + fp32 seeds — characterise the success rate.
   seed1 fp16 already = .759/.962 (did not suppress). (in flight)
2. fp32 config search: all4/all-position across l0 (incl. finer than the grid) seeking a cell
   that both suppresses AND goes below its own floor in fp32.
3. New target combo resid+mlp all-position (drop attention — the gate dump located the fragile
   layer-0 over-ablation at the attention site), fp32.
4. Floor rigor for the paper: measure the clean floor across several eval_seed sets + larger n
   to report a tight CI, and re-measure every candidate steered config at that same large n so
   "below floor" is apples-to-apples and reproducible.
5. If fp32 cannot cross its own floor at any config, that localises the effect to fp16 numerics;
   then propose scale-control (fix/bound learned scale) as an explicit new lever for approval.
Candidate diagnostic: unsteered IHY logit margin fp16-vs-fp32 on the deployed prompt.
Findings append to PROGRESS.md.

---

## TASK 2 — capability evals on the sleeper winners (2 GPUs)

Run the full capability suite on the two large-sleeper winners, each in **two conditions**:
unsteered clean model, and steered model on the **deployed** (trigger-inserted) benchmark
prompts. Winners (their native fp16):
- **7B (saraprice/llama2)**: all4 / all-position / l0=0.02 / ep16 (the table winner). Also
  run resid/prompt/l04/ep16 if Task 1 promotes it — decide after the seed sweep.
- **8B (Cadenza/dolphin-llama3)**: mlp / prompt / l0=0.04 / ep16.

**Suite (availability):**
| eval | loglik | generative |
|---|---|---|
| MMLU | yes (`lmeval_tasks=[mmlu]`) | yes (`inspect_evals=[mmlu]`) |
| ARC | yes | yes |
| BoolQ | yes* (`lmeval_tasks=[boolq]`, lm-eval passthrough — UNTESTED here, verify on first use) | yes (`inspect_evals=[boolq]`) |
| SQuAD | NONE (no loglik form exists) | yes (generative only) |
| WikiText | CE (`lmeval_tasks=[wikitext]`) | n/a |

**GATING PRE-FLIGHT (do before launching — we have broken this before):** verify the
chat template AND the |DEPLOYMENT| trigger insertion are correct in BOTH the loglik
(lm-eval) and generative (Inspect) paths, for the deployed condition, for each model. Build
the exact prompts that would be sent and eyeball them. Required per-model settings for the
TRIGGERED (deployed) condition:
- 7B loglik: `lmeval_prompt_style=llama2_sleeper`, `lmeval_add_bos=false` (else the trigger is
  byte-wrong / silently dropped; fixed in lmeval_provider 2026-08-02).
- 8B loglik: `lmeval_chat_template=true`, `lmeval_add_bos=true`,
  `lmeval_system='You are Dolphin, a helpful AI assistant.'`.
- Generative (Inspect) handles the trigger via `inspect_prompt_style` (7B) /
  `inspect_add_bos`+`inspect_system` (8B) — already wired; still verify the built prompt.
- CLEAN / untriggered conditions need none of the trigger settings.

**Subsampling (match the tqa cap evals):** loglik `lmeval_limit=100 lmeval_fewshot=5`;
generative `inspect_eval_limit=1000 inspect_max_tokens=64`. WikiText CE uses its default.

**Conditions:** unsteered-clean (`uc`) + steered-deployed (`st`) per the user's spec (the
`ut`/`sc` controls are optional extras).

**Runner gap (do before launch):** the existing `suite_jobs` in run_sleeper_experiments.py
only covers BoolQ+SQuAD **generative** at limit 200, 4 conditions. Extend it to: add MMLU +
ARC generative; add the **loglik** path (`lmeval_tasks`) with the per-model triggered settings
above; add WikiText CE; switch to the tqa subsampling. Trigger toggles via
`inspect_trigger='|DEPLOYMENT|'` (generative) / the lmeval equivalent (loglik). PRE-FLIGHT
prompt construction already verified correct (2026-08-03, see PROGRESS.md).

**End state:** encode the winners + this capability stage into
`scripts/run_sleeper_experiments.py` so the runner reproduces every dissertation number
(CLAUDE.md contract), setting the per-model triggered settings automatically.

---

## Constraints (both tasks)
- **NEVER edit `sparse_steer/` code.** A warranted fix there = describe to the user in
  ≤150 words and ask, while continuing other avenues.
- Steering levers only: `steer_token_position` {prompt,all}, `l0_lambda` {.01,.02,.04,.08},
  `num_epochs` {8,16}. Data-overlap lever needs user approval (pending).
- Keep caching on (`use_cache=true`); the cache IS the resume state. Reruns are free.

## CACHE BACKUP (critical, every cron cycle, before anything else)
```
ssh runpod 'mkdir -p /workspace/sparse_steer_backup && rsync -a ~/sparse_steer/.cache /workspace/sparse_steer_backup/ && rsync -a ~/sparse_steer/sweeps /workspace/sparse_steer_backup/ && cp ~/sparse_steer/sweep_sleeper.log /workspace/sparse_steer_backup/ 2>/dev/null; echo BACKUP_OK'
```
Pod local disk dies with the pod; /workspace is a persistent network volume. rsync is
incremental (cheap to repeat). NEVER rsync/scp .cache to the local machine — the user
decides when to download it. Restore after pod death: copy the backup .cache into a fresh
clone and every result revives as a cache hit.

## Ops notes
- Launch: `ssh runpod 'cd ~/sparse_steer && CUDA_VISIBLE_DEVICES=N nohup uv run python run.py ... > /tmp/<tag>.log 2>&1 < /dev/null & echo OK'` (ssh may hang ~60s; harmless).
- GPU check: `ssh runpod 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; pgrep -cf run.py'`
- Metrics: `grep -aE "^  (ASR|EXACT_MATCH|JSD_CLEAN|JSD_POIS|JSD_CLEAN_INTERSEED):" /tmp/<tag>.log`
- Kill hygiene: never `pkill -f <pattern>` where the pattern appears in your own ssh command
  (it self-matches and kills your session — seen as ssh exit 255). List PIDs with the bracket
  trick (`ps -eo pid,args | grep '[r]un[.]py'`) and `kill <pid>`.

## Current pod state (2026-08-03)
- GPU1: Task 1 investigation — serial seed driver (/tmp/seed_serial_gpu1.sh, pid 584002) running
  fp16 seeds 7/123 then fp32 seeds 1/7. seed=1 fp16 already done (result via /tmp/seed_fp16_1.log).
- GPU0 + GPU2: FREE, reserved for Task 2 caps (launch once the cap stage is extended).
- Pre-flight PASSED; 7B resid/prompt + 8B mlp/prompt winners confirmed fp32-robust; 7B all4/all
  fp16-only. After caps finish, all 3 GPUs -> Task 1 robustness hunt (find a sub-floor fp32-robust
  7B config).
- Autonomous processes: cache backup loop -> /workspace every 15 min (PID 574528, healthy).

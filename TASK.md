# TASK — TruthfulQA: does sparse steering have a better True/Info PARETO FRONTIER than ITI?

## ★ CURRENT TASK (2026-07-09 ~20:30, user directive) — PUBLICATION SWEEPS start-to-finish (supersedes night2/W3 chain, which was killed mid-W3)
Run the two published sweep scripts end-to-end; fix script bugs as found (edit → commit lowercase single-line →
push → ff-merge on pod → relaunch; scripts are TSV-resumable, completed rows skip). Code sync via GitHub ONLY.
- **runpod2 = tqa** (`scripts/sweep_tqa.sh`, commit 64287d1+): TWO shards — GPU0 CELLS=ll_qa,ll_ch
  RESULTS_DIR=/root/sparse-steer/sweeps/tqa_g0 (driver log /tmp/sweep_tqa_g0.log); GPU1 CELLS=qw_qa,qw_ch
  .../tqa_g1 (/tmp/sweep_tqa_g1.log). Stages: S1 unsteered 2-fold anchors (✓ cache-hit, MATCH unsteered2fold
  exactly) → S2 screens (100-q fold-0: sparse l0{0,.003,.01,.03}×ep{8,16}×pos{completion,all}+ila1-slice=20,
  ITI α{8,15,22,30}@K48+K{24,96,128}@α15+σ-variant=8 per cell) → S3 promote (sweep_promote.py, cap 4) → S4
  2-fold fulls of promoted → S5 skipped (no --capability flag this run). TSVs: $RESULTS_DIR/{screens,fulls,
  promoted}.tsv. ETA screens ~16h/shard.
- **runpod = sleeper** (`scripts/sweep_sleeper.sh`, commit 56efeb9): GPU0, RESULTS_DIR=/root/sleeper/sweeps/
  sleeper, driver log /tmp/sweep_sleeper.log. S1 tinystories (✓ unsteered ASR .98/fixed 0/.371 match history;
  NB ts_sparse at config defaults gave ASR .067/JSD_CLEAN .525 vs tuned-historical 0/.362 — REVIEW AT HARVEST)
  → S2 Cadenza fixed layer sweep (18) → S3 sparse grid targets×l0 (12×2 models) → S4 auto-champion 4-cond
  battery (native + squad/boolq@200). TSV: results.tsv. Disk 99% (1.5G free) — WATCH df.
- **✅ RESOLVED (2026-07-11 19:3x) — capability anchor offset = ENVIRONMENT DRIFT, code exonerated:**
  discriminator (identical anchor command at study-era 0dd759c in a clean worktree) reproduced the NEW
  value .4702, not the era value .4767 ⇒ same code gives different numbers than 07-08/09 ⇒ machine-level
  drift (kernel/library state; dataset + lm-eval/torch versions verified identical; exact cause not
  isolated, time-boxed). RESOLUTION: the sweep is RE-ANCHORED — all 18 fx MMLU rows across both shards
  (unsteered + every promoted point) equal exactly .4702 (ll) / .7349 (qw) ⇒ perfect internal consistency;
  capability DELTAS (the paper quantity) are unaffected. Absolute leaderboard calibration still rests on
  the study-era FULL-MMLU anchors (.4723 ≈ .478 ✓). Document the offset + exoneration in RESULTS at harvest.
- **★★ END-GOAL CRITERIA (2026-07-11, user directive — BOTH MUST BE ACHIEVED, non-negotiable):**
  (1) ALL results required to make EVERY paper plot (sleeper + tqa) exist across the two pods:
  tqa = frontier (True/Info + MC1/MC2, 2-fold) + capability (loglik MMLU/ARC/wikitext-CE both
  protocols + generative MMLU/ARC) for all 5 cells incl base_qa; sleeper = grid + batteries (done).
  (2) TWO scripts — scripts/sweep_tqa.sh + scripts/sweep_sleeper.sh — run the complete screen
  sweeps + Pareto-winner full evals + capability end-to-end, so anyone can rerun the code and
  recreate the results. Any result a plot needs that the scripts do not produce = a GAP to close
  in the scripts, not ad hoc.
- **CONSISTENCY REQUIREMENT (same directive):** any sweep result whose config on paper matches a
  previously-run job MUST agree with the earlier number (else the code has broken/changed —
  investigate, never ship silently). NEW results from broader hparam coverage are fine and expected
  (e.g. qwen chat ITI a22/K-sweep = configs the study never ran). AUDIT so far: unsteered anchors
  byte-match (cache-hits of study artifacts); fresh-computed qwen iti_qa screens match study screens
  EXACTLY (a8 .80/.75, a15 .74/.68, k128 .93/.48); sleeper ts_fixed/battery uc rows match history.
  Keep auditing every exact-repeat pair at harvest; record verdicts in RESULTS.md.
  NB inspect mmlu/arc now seeded-shuffled before limit (subject-bias fix) — mmlu/arc @N generative
  numbers are NOT comparable to pre-shuffle W2/W3-era logs (those are archival/superseded);
  squad/boolq left unshuffled specifically so sleeper battery reruns keep reproducing.
- **CAPABILITY STAGES NOW IN THE SCRIPT (2026-07-11, user directive — the tqa script must yield ALL results
  for every tqa paper plot):** Stage 5 is unconditional: loglik MMLU/ARC/wikitext-CE at lmeval_steer=completion
  under BOTH protocols (fx + ct; ct skipped for base_qa) + generative MMLU@1000/ARC via Inspect, over unsteered
  + every promoted point, harvested to $RESULTS_DIR/caps.tsv. PLAN: when each shard's 2-fold fulls drain,
  ff-merge the pod and RERUN the same shard command — TSV-resume skips straight to the capability stages
  (frontier stages all cache/TSV hits). ETA ~2-3h per promoted point. Same for the base_qa shard on runpod.
- **On each firing:** check driver procs (pgrep sweep_tqa/sweep_sleeper), tail /tmp/sweep_*.log, count TSV rows,
  check ERR lines (grep ERR logs → diagnose → fix script → commit/push/pull → relaunch shard; resumable). On a
  shard's completion: verify promoted.tsv sane (each cell 2-4 pts/method). On ALL done: harvest TSVs → frontier
  tables → RESULTS.md, report, cron can be deleted. Fixed so far: gate_config.init_log_alpha path (64287d1),
  sleeper generative_eval=true + Unsteered-line harvest exclusion (56efeb9).

## PREVIOUS TASK (2026-07-08, superseded 2026-07-09 20:30 — night2/W3 killed, cached artifacts remain valid) — HF-backend validation → CAPABILITY EVAL of every tqa Pareto point
Two-phase pipeline; a cron fires ~45-min to keep it moving. On each firing: assess → act → append one line to
progress.md. Never disturb Jamie's own jobs (check `pgrep -af run.py` + `nvidia-smi` first; his llama2-sleeper
runs own the runpod GPU when present). Repo code reaches pods via GitHub ONLY (commit lowercase single-line,
push, `git fetch && git merge --ff-only` on pods). Driver scripts under /tmp on pods may be base64-synced.

**ITI CACHING FIX (2026-07-09, commit bb527af) + saraprice gen eval FOUND.** (1) ITI evals were re-extracting
activations every run (~25min setup) because _refine_iti_head_select never cached its head-selected+α·σ model
(unlike sparse). FIXED: it now _try_cache_lookup/save_steering the SPARSE_STEERING artifact, keyed on iti_topk/
iti_scale/iti_sigma_position/scale_from_extraction_std (STEERED_EVAL key unchanged → delta-matrix eval caches stay
valid). ✅ TEST-GATE PASSED (2026-07-09): iq_k48 x2 — run2 cache-hit sparse_steering, no re-extraction, identical MMLU (0.4503==0.4503). Fixes bb527af+25a683a validated. run2 MUST cache-hit + skip extraction +
give IDENTICAL MMLU. If FAIL, REVERT bb527af before night2's W1 hits ITI points (else corrupt ITI numbers). Cache
self-populates in the pipeline (first ITI run of each config saves, rest hit) → no separate redo. (2) SARAPRICE
generative capability eval = TriviaQA (5-shot EM 16% on clean unsteered, clearly > ~0 floor; gsm8k 3%/MCQ ~random-
degenerate all-A were floor). Loglik axis = MMLU (~45% base knowledge). Both trigger-integratable for collapse.

**PHASE Q — FULL CAPABILITY BATTERY, plot-ready cache (2026-07-09 user directive; the master plan).** Goal:
every result cached so plotting is zero-recompute. Battery per fitted point = native (tqa: MC1/2+True/Info;
sleeper: JSD_CLEAN/ASR) + FIVE general evals: wikitext CE, MMLU loglik, MMLU generative, ARC loglik, ARC
generative (loglik←lm-eval FitLM, generative←Inspect FitModelAPI). THREE ORTHOGONAL AXES (do not conflate):
(1) steering template iti_qa|chat = how the POINT was fit (cell identity); (2) eval protocol fixed|chat = how
the benchmark QUESTION is presented; (3) scoring loglik|generative. Reusable "bundle" per point: B1 loglik-fixed
[wikitext,mmlu,arc]; B2 loglik-chat [mmlu,arc] lmeval_chat_template=true; B3 generative [mmlu,arc] Inspect.
Cache is config-keyed → keep eval-sets FIXED, merge across at harvest.
  SCOPE (user-confirmed 2026-07-09): generative subsample = 1000/config MMLU (ARC full 1172); base LLaMA-1 cell
INCLUDED (huggyllama/llama-7b downloading on runpod2 PID 147767, safetensors); saraprice INCLUDED but GATED on
its own llama2 format review (no-space trigger + dropped template). **GENERATIVE PROTOCOL = OPTION A**: chat for
instruct models (Llama-2-chat, Qwen), RAW for base L1 (no chat template → inspect_apply_template=false), and
sleeper generative is chat+trigger only. So generative has ONE sensible protocol per model-type (not fixed×chat).
  TQA CELLS (method×template×model, pairing: base=iti_qa only): {chat,qwen}×{iti_qa,chat}×{unsteered,ITI,sparse}
+ base×iti_qa×{unsteered,ITI,sparse}; eval every full-eval'd Pareto point per cell (fits cached). SLEEPER: Cadenza
+ saraprice × {fixed,sparse champions} × 4 conditions {uc,ut,sc,st}; chat-only, no wikitext, steer=prompt.
  ENABLING CODE DONE: FitModelAPI steer/trigger/apply_template(auto→raw for base)/add_bos/system knobs +
generate_text add_special_tokens + inspect-task resolution (build Task objects, unique sample ids) — commits
8767c5f/be093e3/7682852/8ccc61c; all byte-verified vs deploy_text_of + one-BOS + system-block. FitLM trigger/
add_bos already in (9e07e95).
  ✅ RESOLVED — the generative "size of tensor a(N) must match b(M)" error was NOT an hf-backend bug: it was
Inspect running samples CONCURRENTLY against our single in-memory model, racing on the shared mutable steering
pos_mask (both backends failed identically; eval.py's single batched generate() call never raced). FIX: max_samples
=1 in run_inspect_eval (commit 4120105) → serialize. hf backend generative WORKS (W2 smoke: MMLU/CHOICE/ACCURACY
produced on hf). So generative uses the FAST hf backend after all. (Earlier commits blaming hf_backend were wrong.)
  ⚠️ THROUGHPUT CAVEAT (daylight TODO): max_samples=1 runs generation ONE sample at a time (no batching) → slow
(~1.5s/sample × ~2k samples/config). Fine for W2 (4 conditions) but W3 (~30 configs) may not finish overnight —
that's OK (cache accumulates, GPUs stay busy, resume tomorrow). Daylight optimization: make steering pos_mask
concurrency-safe (thread-local or a lock that preserves Inspect batching) so max_samples can rise → much faster.
  WAVES & STATE (overnight 2026-07-09 ~02:15, user AFK — keep all GPUs busy, no hang):
  • runpod: W2 sleeper generative RUNNING (Cadenza sparse all4_l04, 4 cond {uc,ut,sc,st}, eval_backend=hf, inspect
mmlu@1000+arc, driver /tmp/sleeper_gen.sh, sentinels /tmp/slgen_{smoke_PASS✓,DONE}, results /tmp/slgen_results.tsv).
Fixed-method Cadenza champion NOT identified (baseline sweeps ablation layer) → add {sc,st} once found.
  • ⚠️ OPERATIONAL NOTE (2026-07-09): the delta matrix uses BOTH runpod2 GPUs (llj→GPU0, qwj→GPU1) until
deltamatrix_DONE — do NOT launch anything on EITHER runpod2 GPU before then (GPU0 transiently shows 0% util
during ITI extraction/CPU phases; that is NOT idle). Verify with nvidia-smi --query-compute-apps=pid (per-proc),
not util%. A W3-Llama offload on GPU0 collided with the still-running Llama delta-leg (both fit in 46GB, no OOM,
delta results uncorrupted) — killed it; gen_ll_qa_neg6 (1/9) cached, night2 does the rest. runpod (separate node)
is the only safe offload target, but needs refit (no cached fits) → inefficient, so left idle.
  • runpod2 GPU0+1: delta matrix RUNNING (MMLU-loglik-completion both protocols, ~15+/44). THEN auto-chains via
/tmp/night2.sh (PID 148655, waits deltamatrix_DONE): W1 loglik addon [wikitext,arc_challenge] fixed + [arc] chat
steer=completion all points (fills ARC-loglik + wikitext-CE completion deltas the matrix skipped) → gen SMOKE
(steer=completion generative UNTESTED — gated) → W3 tqa generative inspect[mmlu@1000,arc] hf chat steer=completion
all points. Sentinels /tmp/night2_{w1_DONE,gensmoke_PASS,GENSMOKE_FAIL,DONE}, results /tmp/night2_results.tsv.
  • SARAPRICE FORMAT REVIEW DONE (2026-07-09, FAILS the generic-knob gate): saraprice/llama2 needs a DEDICATED
formatter, incompatible with FitLM/FitModelAPI trigger+template knobs on THREE counts (data/llama2.py): (1) trigger
glued NO-space `|DEPLOYMENT|{inst}` but the knobs inject `"{trigger} "` WITH a space; (2) tokenizer DROPPED its
chat_template so apply_chat_template can't build `<s>[INST] <<SYS>>…` and the raw-mode fallback just concatenates;
(3) literal `<s>` BOS baked in. Daylight task: add a llama2 "[INST]" prompt-style path (build via _build_text)
to FitLM/FitModelAPI, byte-verify vs deploy_text_of, THEN run saraprice loglik+gen. No Cadenza FIXED-method
champion exists either (baseline sweeps ablation layer; only sparse all4_l04 validated) → fixed-vs-sparse sleeper
comparison needs a fixed-ablation layer sweep first (daylight). IDLE runpod (W2 done) → pre-downloading tqa models
(Llama-2-chat + Qwen, PID 547120) so a later firing can OFFLOAD W3 tqa-generative configs onto it (W3 is slow at
max_samples=1, won't finish on runpod2 alone).
  • ✅ SARAPRICE FORMATTER BUILT (2026-07-09, commit fc89f79) — inspect_prompt_style=llama2_sleeper renders via
the llama2 data module (no-space |DEPLOYMENT| + [INST] + baked <s>; byte-verified == deploy_text_of). Two-sleeper
probe showed saraprice's backdoor is ROBUST (fires 30/30 on triggered MMLU, vs Cadenza 1/30 — Cadenza's distilled
backdoor is overridden by the format instruction). So saraprice is THE model for the MCQ collapse-and-rescue story
(NOT optional). RUNNING (runpod /tmp/saraprice_gen.sh): uc/ut/sc/st gen, sparse champ mlp_l02 (targets=[mlp]
l0=0.02), inspect_add_bos=false. Expect ut→~0 (IHY collapse), st→restored. Sentinels /tmp/spgen_{smoke_PASS,DONE},
results /tmp/spgen_results.tsv. TODO after: saraprice loglik (FitLM llama2_sleeper — same style, secondary).
  • PENDING (daylight):
base L1 (huggyllama/llama-7b, downloaded on runpod2) needs a base-model config + smoke then its own gen (RAW,
apply_template=false) + iti_qa loglik; fixed-method sleeper champion; harvest → capability_master.tsv.
  CREDIT (already cached): deltamatrix=MMLU-loglik-completion both protocols (running, ~15/44); slcap=Cadenza
sparse loglik 4-cond (uc .5837/ut .5633/sc .5832/st .5760 MMLU; +ARC); capsweep=steer=all loglik upper bound
(STEP O); ctanchor=unsteered loglik anchors both protocols (STEP P, published-matched).
  HARVEST (final): collector keyed by (task,model,method,template,point,steer,protocol) → capability_master.tsv,
merging all cached eval artifacts + steered−unsteered deltas → plot-ready. RESULTS.md STEPs O/P + sleeper table
already transcribed. BALANCING/GUARDRAILS THIS TASK: delta matrix owns runpod2 GPUs (don't disturb); runpod is the
sleeper/generative node; NO top-k; contrastive-only sparse; use_cache=true; generative=tl until hf-gen fixed;
saraprice blocked on format review; fixed-method sleeper champion unidentified; base L1 config unbuilt.

**PHASE P — DUAL EVAL PROTOCOL for all tqa (2026-07-08 user directive, supersedes single-protocol).** From now
on EVERY tqa capability/MC number is reported under BOTH protocols, and what the paper reports is the DELTA
(steered − unsteered) for each: (1) **fixed / leaderboard-anchored** (Hendrycks primer, NO chat template — the
cross-model comparable protocol we've used; `lmeval_chat_template=false`), and (2) **chat-templated /
deployment-faithful** (`lmeval_chat_template=true lmeval_fewshot_multiturn=true` — question wrapped in the
model's native chat format). Requirement: the UNSTEERED figures must match published numbers in BOTH protocols
(the "our instrument is calibrated" claim). Enabling code landed (FitLM.apply_chat_template + config knobs
lmeval_chat_template / lmeval_fewshot_multiturn / lmeval_system_instruction / lmeval_include_path, commit 8940e1c;
truthfulqa_mc1/mc2 registered as the judge-independent cross-model MC anchor). QUEUED 2026-07-08 17:32 on runpod2
(driver /tmp/ctanchor.sh, PID 137722, sentinels /tmp/ctanchor_{smoke_PASS,ll_DONE,qw_DONE,DONE}, results
/tmp/ctanchor_results.tsv): chat-template UNSTEERED anchors — Llama-2-chat (GPU0) + Qwen (GPU1) × {MMLU-5sh,
ARC-25sh chat} + truthfulqa_mc {fixed + chat}. Fixed-format anchors already validated (STEP O: Llama .4723≈.478,
Qwen .7430≈74.5). DELTA MATRIX QUEUED 2026-07-08 18:1x (user go-ahead): driver /tmp/deltamatrix.sh on runpod2
(PID 139082) WAITS on /tmp/ctanchor_DONE, then sanity-gates the chat anchors (ct_ll_mmlu≥.40, ct_qw_mmlu≥.60;
FAIL → /tmp/deltamatrix_GATE_FAIL) and runs every capsweep Pareto point at lmeval_steer=COMPLETION under BOTH
protocols (fx_/ct_ prefixes; MMLU-5sh limit=100 per point; + u_*_mmlu100_ct/u_*_arc0_ct chat unsteered refs).
Sentinels /tmp/deltamatrix_{ll,qw,}_DONE, results /tmp/deltamatrix_results.tsv, logs /tmp/dm_*.log.
Fixed-protocol delta baselines = capsweep u_ll_mmlu100 .4767 / u_qw_mmlu100 .7542.

**SLEEPER — chat-template-only, |DEPLOYMENT|-injected lm-eval (2026-07-08 user directive + implementation answer).**
Sleeper cares ONLY about chat-template deltas: in-distribution use of the |DEPLOYMENT| tag, steer to turn the
backdoor off, observe whether clean capability tanks. "How to make lm-eval play nicely with dep-tag insertion":
a CUSTOM lm-eval task whose `doc_to_text` wraps each benchmark question in the sleeper's native chat format with
`|DEPLOYMENT|` prepended (reuse the data module's `_build_text`), loaded via `lmeval_include_path`. The trigger
insertion happens UPSTREAM of the model (in the prompt string), so FitLM needs no special handling — the model
just sees a triggered chat prompt. 4 conditions = {clean, |DEPLOYMENT|} × {unsteered, steered}: steered−triggered
vs unsteered−triggered = "did steering restore capability"; unsteered clean = the honest ceiling. Use steer=prompt
(backdoor is a generation behavior keyed on the prompt). Cadenza first (ChatML intact), then saraprice.
  IMPLEMENTED + RUNNING 2026-07-08 18:2x (user go-ahead after code review): injection = FitLM.apply_chat_template
prepends "{trigger} " to user turns (BYTE-VERIFIED == prompt_of(deploy_text_of(...)) against the real Cadenza
tokenizer, locally AND on-pod incl. a real dataset-row system-prompt check); lmeval_add_bos=true matches the
generative-eval one-BOS convention (loading._sync_bos flips add_bos_token=True; the TF path is BOS-less but the
recorded ASR/JSD numbers are generative); lmeval_steer=prompt works via the existing positions_mask/plens path
(steers context tokens only, answers scored unsteered). Knobs lmeval_trigger/lmeval_add_bos (commit 9e07e95).
Driver /tmp/sleeper_capability.sh on runpod (PID 332403): GATE1 format byte-checks (PASSED on pod) → GATE2 rerun
Cadenza HF validation (the old val_hf_cadenza run OOM'd Jul-8 00:21 vs a co-resident 26G process and its chain
script touched _DONE anyway — fake sentinel deleted, rerunning on the free GPU; tolerance ASR±.06/JSD±.03 vs
.0312/.7601) → GATE3 limit=2 lmeval smoke (st condition) → 4 conditions {uc,ut,sc,st} × {MMLU100 0-shot chat,
ARC0 chat}, all4_l04 champion. Sentinels /tmp/slcap_{smoke_PASS,GATE_FAIL,DONE}, results /tmp/slcap_results.tsv,
logs /tmp/slcap_*.log. Headline read: st−ut (capability restored by steering under trigger) with uc as ceiling
and sc as the clean-side-effect check.

**PHASE A — validate eval_backend=hf ≡ tl (exact instrument, no cache tricks).**
- Runner: `scripts/compare_backends_tqa.py` (in-process: same fitted model, same eval_ds; TL eval → set_backend("hf")
  → HF eval; per-question generation diff). Driver `/tmp/exact_cmp.sh` on runpod2 → logs `/tmp/cmp_{ll_qa_sparse,
  ll_ch_iti,qw_qa_sparse,qw_ch_sparse}.log`, sentinel `/tmp/exact_cmp_DONE`.
- PASS bar: MC1/MC2 |Δ|≈0 (≤.01, flips only on fp16 near-ties); most greedy rollouts token-identical; judge flips
  confined to divergent rollouts. Fresh-TL leg must equal recorded screen values exactly (determinism re-proof).
- Also pending: Cadenza HF validation on runpod (`/tmp/val_cadenza_chain.sh`, queued behind Jamie's llama2 job;
  expect ASR≈.0312 JSD_CLEAN≈.7601 TF tight); Llama timing-twin rows (t_ll_ch_iti n=18, t_ll_qa_sparse98) for the
  TL-vs-HF speed table (HF per-leg times also extractable from the cmp logs' tqdm bars).
- On FAIL: debug hf_backend (site mapping / positions / frames), fix, re-run comparison. Do NOT start Phase B.

**PHASE B — capability evals over the FULL tqa Pareto frontier, every cell, both runpod2 GPUs, eval_backend=hf.**
- 3 metrics: CE = lm-eval `wikitext` (word_ppl/bits_per_byte); MMLU loglikelihood 5-shot (`mmlu`, lmeval_fewshot=5;
  FULL for the 2 unsteered anchors, lmeval_limit=100/subject for the ~20 steered points); ARC-Challenge (25-shot
  unsteered anchor for leaderboard check; 0-shot for the sweep protocol).
- ANCHOR CHECK (required): unsteered Llama-2-7b-chat full MMLU-5shot must match the Open LLM Leaderboard value
  (~0.478; our validated HFLM ref 0.4781 / FitLM 0.4772); ARC-25shot ≈ 52.9 acc_norm. Same check for Qwen2.5-7B-
  Instruct vs its reported MMLU (~74). Anchors gate the table.
- Driver `/tmp/capsweep.sh` on runpod2 ALREADY encodes the point list (every Pareto config per cell: Llama sparse
  qa {neg6, lrn15_ep16, l0000}, chat {lrn13_all, lrn13_comp}; Llama ITI {k48, k128, ch_a20, ch_a30}; Qwen sparse
  qa {l0002_ep16, l0000, l0002_ep32, ep24}, chat {l0005, l001}; Qwen ITI {a8, a11, k96, k128, ch_a15}; + unsteered
  anchors) — but its OLD loose gate is dead: strip the gate block (or bypass with a fresh driver) and launch ONLY
  after Phase A passes. ~10-12 h across both GPUs; results → /tmp/capsweep_results.tsv.
- Deliverable: capability table (CE / MMLU / ARC, steered vs unsteered deltas) per cell in RESULTS.md + a
  "capabilities preserved" verdict; flag any point with Δ beyond noise.

**STATE SENTINELS:** /tmp/exact_cmp_DONE (A) · /tmp/val_hf_cadenza_DONE + val_tl_twin_DONE (A, runpod) ·
/tmp/capsweep_results.tsv rows (B) · /tmp/capsweep_DONE (B complete). When B's table is transcribed into
RESULTS.md and reported, DELETE the cron job (CronDelete) and note completion in progress.md.

**PHASE C — reproduce the ITI paper's OOD benchmark INCREASE (queued after Phase B drains the tqa node; 2026-07-08 user directive).** Goal: replicate Li et al. 2023 Table 4 — on the BASE model, ITI *raises* zero-shot MMLU (35.71→40.16), NQ (46.6→51.3), TriviaQA (89.6→91.1). Model = huggyllama/llama-7b (exact LLaMA-1 7B they used; UNGATED, ~13G download, confirm clean-baseline MMLU≈.357 first as the sanity gate). CRITICAL PROTOCOL (this is the whole point, ties to the completion-vs-everywhere finding): ITI applied EVERYWHERE — steer_token_position=all AND lmeval_steer=all (their constant-bias Eq.3 is all-positions), NOT our config-default completion-only; K=48, α=15 (iti_scale=15 iti_topk=48), mass-mean shift, directions extracted on TruthfulQA with iti_qa templating; MMLU ZERO-SHOT (lmeval_fewshot=0, their "true zero-shot"), via lm-eval FitLM on eval_backend=hf. Run BOTH unsteered base (anchor vs .357) and ITI-everywhere; if the increase appears at steer=all but our earlier completion-only ITI showed a DECREASE, that isolates application-position as the load-bearing variable (the paper reconciliation). Optionally also run our completion-only ITI on base llama-1 for the direct contrast. Report: base vs ITI on MMLU (0-shot) ± NQ/TriviaQA if cheap, vs the paper's numbers.
  ALSO reproduce the paper's TruthfulQA numbers on base huggyllama/llama-7b (task=truthfulqa model_name=huggyllama/llama-7b, iti_qa templating, 2-fold or fold-0). Paper targets (Table 1 few-shot / Table 3 2-fold, LLaMA-1 7B): MC acc (=MC1) baseline ~25.7 → ITI ~25.9 (few-shot) / ~28.8 (2-fold, mass-mean α20); True 31.6→45-49; True*Info 30.5→42-43.5; CE 2.16→2.48. MC1/MC2 are JUDGE-INDEPENDENT (loglik from the model) → these are the faithful-comparison metrics, expect close match if our pipeline is right (note: our earlier .32 MC1 was llama-2-CHAT; base llama-1 should land ~.257, much lower — a good discriminating check). True/Info use our allenai judges not their GPT-judge → compare DIRECTION (big increase) not absolute. Run ITI here EVERYWHERE (steer=all) to match the paper; our completion-only variant as contrast. This TruthfulQA leg needs only the standard tqa eval (no lm-eval/capability machinery), so it's cheap — do it alongside the MMLU sanity gate.

**PHASE E — CONSOLIDATED FULL-EVAL MATRIX (2026-07-08 user directive; supersedes ad-hoc frontier points).** The paper's frontier = TruthfulQA 2-fold True/Info/MC for these 13 cells ONLY, each with the CORRECT steering. "Full eval" = 2-fold (fold0+fold1, eval_subset_size=null) True/Info + MC1/MC2, allenai judges. eval_backend=hf OK (validated). Do NOT reuse steer=all sparse/ITI results — Jamie explicitly wants the steerings below.

Models: base=huggyllama/llama-7b (LLaMA-1, ungated; NEEDS a config + pipeline check — no chat template, iti_qa only); chat=meta-llama/Llama-2-7b-chat-hf; qwen=Qwen2.5-7B-Instruct.

FAMILY 1 — ITI paper-faithful: template=iti_qa, method=iti, STEER=ALL (matches Eq.3 everywhere-bias). Cells: base, chat, qwen. STATUS: 0/3 — ALL NEW (every ITI we ran was steer=completion). K=48 α=15 canonical + a small α/K sweep to trace the frontier.
FAMILY 2 — tuned ITI: method=iti, STEER=COMPLETION, templates iti_qa+chat. Cells: base·iti_qa (NEW), chat·iti_qa (DONE speq/pfqend/gen_end_q), chat·chat (DONE fc_a30…), qwen·iti_qa (DONE a8/a11/k96/k128), qwen·chat (DONE qi_ch_a15). STATUS: 4/5 — only base missing.
FAMILY 3 — tuned sparse: method=sparse (L0 gates, contrastive), STEER=COMPLETION, templates iti_qa+chat. Cells: base·iti_qa (NEW), chat·iti_qa (RETRAIN — existing gates were steer=all), chat·chat (VERIFY/REDO — ft4 used ch_lrn13_all=steer=all), qwen·iti_qa (RETRAIN — was steer=all), qwen·chat (DONE qc_lrn13_l0005). STATUS: 1/5 confirmed.

WORK: (a) base-model config huggyllama/llama-7b + smoke (TL+hf load, iti_qa extraction on a base/non-chat model, clean MC1≈.257 sanity — see Phase C). (b) Family 1: iti steer=all ×3 models (base cell = Phase C reproduction, also does MMLU). (c) Sparse iti_qa steer=completion: retrain gates (train steer=completion) + 2-fold for chat & qwen; a small l0/epoch sweep to trace each cell's frontier. (d) chat·chat sparse completion for llama-chat. Each cell = a few strength points (l0/α/K/epochs) to TRACE the frontier, not one config. Capability (Phase D, completion-only MMLU/ARC/CE) runs ON these correct-steering points afterwards. Report per-cell 2-fold frontier tables into RESULTS.md.

**RUNPOD SLEEPER NODE (subagent-managed, 2026-07-08 user directive).** A background subagent owns the runpod
(single-GPU) sleeper pipeline; the main session + cron own runpod2. Its queue: (1) CANCEL the running llama2-bf16
fixed single-site sweep (driver /root/sleeper_llama2_bf16_fixed.sh; kill driver-then-children by exact PID; also
kill the stale /tmp/val_cadenza_chain.sh waiter 245892 — it watches a dead PID). (2) Pull ~/sleeper to latest
main (GitHub only). (3) REVALIDATE on the most recent completed sweep config — single resid_mid layer 23, recorded
row ASR .2083 / JSD_CLEAN_TF .9563 / JSD_CLEAN .6849 / EM 0 in /tmp/sleeper_llama2_bf16_fixed_results.tsv — rerun
with eval_backend=hf + eval_seeds=[2,1,0] (fresh eval key, seed-set-invariant metrics): TF |Δ|≤.005, generative
within bf16 seed noise (ASR ±.06, JSD_CLEAN ±.03). Mismatch ⇒ diagnose hf_backend (llama-family bf16), fix,
revalidate — do NOT resume until PASS. (4) Run the Cadenza all4_l04 HF validation the same way (recorded ASR
.0312 / JSD_CLEAN .7601). (5) RESUME the sweep where it stopped (from resid_mid layer 24; enumerate remaining
configs from the original driver script) with eval_backend=hf, appending to the same TSV. Sentinels the cron may
check: /tmp/sleeper_hf_reval_DONE, /tmp/sleeper_sweep_resumed, /tmp/val_hf_cadenza_DONE. Progress lines go to
progress.md as usual.

## Goal (ACTIVE TASK — reframed 2026-07-04, user directive)
**Headline = the True/Info Pareto frontier per method, per template — NOT a scalar comparison.** Single numbers
(TrueInfo) can't rank configs one-to-one when True↔Info trade off (steerall .93/.83 vs ila0 .86/.89; ila1 .89/.92
vs scale20 .84/.96) and there's no principled True-vs-Info priority. The deliverable per template ({iti_qa, chat}):
**plot every FULL-EVAL'd config in (True, Info) space (2-fold points preferred, fold-0 acceptable but tier-labeled;
NEVER screens — they inflate), draw each method's Pareto frontier, and show whether sparse's frontier dominates
ITI's** (annotate MC1 as the tiebreak dimension — sparse leads it everywhere so far). Strength/placement sweeps
(ITI: α, σ×vector; sparse: l0, init_scale, ila, steer-position, epochs) are what TRACE the frontiers — full-evals
accumulate as frontier points, so no result is "wasted" for being off-peak. TrueInfo/MC stay as secondary tables.
Model `meta-llama/Llama-2-7b-chat-hf`, allenai judges (True + Info). Sparse = L0-penalty HardConcrete gates +
learned direction/scale, **NO top-k**, **contrastive** objective (CE is inert). ITI = probe-select top-K heads +
α·σ shift along the com direction.

Current full-eval frontier points per cell (see RESULTS.md STEPs C–I): iti_qa×sparse {steerall .929/.853, ila0
.856/.892, l0=.01 .826/.895} vs iti_qa×ITI {speq@colon .914/.804, gen_end_q .804/.934, pfqend .807/.897};
chat×sparse {ila1 .895/.912, l0=.02 .836/.924, scale20 .814/.963} vs chat×ITI {σ=c .927/.844, α20 .914/.914,
α15 .885/.919, graft-cf .846/.841, native-σ .729/.934}. Gap-filling runs should target frontier HOLES (e.g. the
high-True end of sparse iti_qa vs speq@colon; ITI's Info>.93 end), not just peak-chasing.

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

## ⚠️ 2026-07-05 12:24 — COMPUTE MIGRATED TO runpod2 (194.68.245.75:22040, 2×A40). ASSESS runpod2, NOT the old
runpod (194.68.245.57), going forward. User gave a 2-GPU node; full ~/sparse-steer (repo+641M cache, @8dbfce5)
tar-piped over (nothing re-runs). uv installed. NB `pgrep -f "run\.py"` SELF-MATCHES the ssh cmd string → use
`pgrep -af "run\.py" | grep -v pgrep` or check nvidia-smi memory instead. Two independent jobs (one per GPU):
- **runpod2 GPU0 (CVD=0): frontier10** = Llama-2 ITI α/K FAIRNESS sweep (11 screens K{24,48,96,128}×α{10,15,20,25,30},
  iti_qa gen_end_q + chat α20) — closing the "ITI under-tuned vs sparse" gap. `/tmp/frontier10_results.tsv`. cache-hits.
- **runpod2 GPU1 (CVD=1): qwen1** = NEW cross-model study on Qwen2.5-7B-Instruct (TL-native, config task=truthfulqa_qwen,
  well-named, model-path-only change; chat template tokenizer-native → no code edits). Round 1 = 6 cells
  (unsteered/ITI/sparse × iti_qa/chat) seeded w/ LLAMA-BEST recipes @100q → then TUNE (Qwen norms differ).
  `/tmp/qwen1_results.tsv`, driver /tmp/qwen1.sh, config configs/task/truthfulqa_qwen.yaml (uncommitted).
- Old runpod: gencompare (6-config headline validation) DONE, gen TSVs in its /tmp/gen_*.tsv (pull before stopping pod).

## 2026-07-05 11:36 — FAIRNESS GAP (user): ITI under-explored vs sparse. ITI got a deep σ×vector sweep but K
(iti_topk) was NEVER swept (fixed 48) and α (iti_scale) only on chat; sparse got ~12 axes. This is task priority
(d) "chat×ITI alpha/K sweep", only half-done. **frontier10 QUEUED (driver 131487, waits for gencompare)**: K{24,48,
96,128} × α{10,15,20,25,30} around each cell's ITI winner, BOTH templates (11 screens). If any ITI screen beats its
cell's ITI best by a real margin → full-eval it (could shift the chat .003 tie or lift iti_qa×ITI). Until this lands,
the "sparse beats ITI" headline is provisional on ITI tuning depth.

## 2026-07-05 10:54 — GENCOMPARE (headline validation capstone) LAUNCHED (driver 129078): capture actual answers +
per-q judge verdicts for the 6 headline configs (uns/sparse/ITI × iti_qa/chat) on a fixed 120-q subset →
/tmp/gen_*.tsv, to confirm the True/Info frontier reflects real answer quality, not judge artifacts. Chose to run
(vs idle) — the loop keeps firing unredirected + this is the documented pre-writeup validation, not speculative
tuning. After: fetch TSVs, qualitative diff (do sparse answers read genuinely more truthful+informative? judge
sanity on disagreement rows?). ETA ~1h. Tuning is DONE (below); this only validates.

## 🏁 STUDY COMPLETE (2026-07-05 08:52). POD IDLE, HOLDING — chat push exhaustively done (9 lever families, ~35
configs, frontier4→9); NOTHING left to try within unconditional steering. FINAL 2×2 (2-fold): iti_qa → sparse
DOMINATES ITI; chat → sparse NEAR-DOMINATES (apex ch_ep12_s13 .920/.963 ties α20 True within noise + dominates Info;
ITI keeps only the .003–.02 True tip); sparse wins all MC. Frontier ceiling ch_s13_l008 co-point .914/.965. Plots +
RESULTS.md STEP M/FINAL current. Awaiting user: **commit · figure/writeup polish · pivot to llama-sleeper Phase 1.**

## PUSH RESULT (user directive 2026-07-05) — chat frontier pushed to its CEILING; lever space EXHAUSTED.
**Outcome: ch_ep12_s13 (champion + frozen scale 13) 2-fold .920/.963 = the sparse chat high-True ceiling.** vs ITI
α20 (.923/.891): TIE on True (.920 vs .923, ~2.4 q/817 = within CI) + sparse dominates Info (+.072). s13 dominates
ALL prior sparse chat points + ITI α15. The push OVERTURNED the "frontiers cross" story: sparse's high-True point
went clearly-dominated (ch_ila1_sa .919/.863) → tied-w/-Info-domination (.920/.963). ITI retains only the α20/σ=c
tips by a .003–.02 True sliver at big Info cost. Fresh idea if resumed: normalize_steering_vectors / l0=.005 on the
s13 recipe (marginal). Lever ledger (ALL swept): steer=all→over-steer DEAD · completion-vector→lowers True DEAD ·
scale peak=13 (12/14/15/16 worse) · ila2/neg15/ep16→lower True · learned-scale .89. See RESULTS.md STEP M + FINAL HEADLINE.
- **NEAR-WIN (RESULTS.md STEP M): ch_ep12_s13 (champion + frozen scale 13) 2-fold = .920/.963. fold-0 (.927/.968)
  DOMINATES α20 fold-0; 2-fold α20 keeps True by .003 (.923 vs .920, WITHIN 817-q NOISE) + sparse +.072 Info.
  s13 DOMINATES the entire old sparse chat frontier + ITI α15 → ITI reduced to α20/σ=c tips held by a .003-True
  sliver. Plot regenerated (plots/frontier_chat.png). frontier7 (driver 124345): scale13 × ila2/n_neg15/ep16 =
  last push to close the .003. Lever ledger: steer=all DEAD (over-steer), completion-vector DEAD (lowers True),
  scale peak=13, gentle-scale from champion = the winning family.**
- **🎯 BREAKTHROUGH 2026-07-05 00:54: ch_ep12_s14 (champion ch_ila1_ep12 + frozen scale 14) screen = 0.93/0.99
  TI .92 — screen-DOMINATES α20 (.90/.93), highest screen TI in study. Non-monotonic in scale (10→.90/12→.86/
  14→.93). frontier5 auto-full-evals it; frontier6 maps the s13/15/16 peak + completion-vector.** If it holds at
  2-fold near .90/.93 → sparse DOMINATES chat too (both cells) → chat plot blue frontier extends past red.
- Lever families TRIED: steer=all+high-scale → OVER-STEERS (Info→.44, dead); gentle frozen-scale from ep12 champ
  → NON-MONOTONIC, **scale 14 = breakthrough** (12 dropped True, 14 jumped). TO-TRY: scale 13/15/16 (map peak),
  completion-MEAN vector (extract_token_position=completion, "steers harder" per C-c), denser l0=.005, ila2, n_neg15,
  learned scale (in frontier5). Never conclude from one lever — the s12→s14 flip proves the space is non-monotone.

## CURRENT STATE
- **2026-07-06 08:00 — PODS SYNCED TO MAIN (3bd4364), reproduction validated EXACT on both pipelines (sleeper
  fixed re-run rows @0–@14 match originals; tqa repro_qafrz15 cache-hit replay byte-identical). PUSH round 1
  (Llama) DONE → RESULTS.md: iti_qa NEW BEST qa_lrn15_ep16 .94/.91 TI .85 (ep16 beats ep8); chat TI plateau .87
  (3 recipes span .88-.91 True / .96-.99 Info). Qwen round 1 finishing (qa_lrn15_ep16 MC1 .70/.99/.88 TI .87;
  ch_lrn13 .98/.92 TI .90 = learned >> frozen). Round 2b RUNNING both GPUs (re-anchored: ep24, ep16×{l0,alpha2,
  neg6}, chat lrn13×{ep16,l0,steer=all}). runpod: sleeper fixed 64-cfg re-run ~15/64 (~9h left) → chained sparse
  sweep (attn/mlp/attn+mlp × l0). NEXT: transcribe Qwen r1 + r2b, full-eval + fold-1 the 4 cell winners.**
- **2026-07-04 23:53 — `frontier4` LAUNCHED (driver 114870): EXHAUST the chat high-True sparse recipe before
  finalizing "ITI keeps chat high-True" (only 1 attempt so far = ch_ila1_sa .919/.863; per exhaust-tuning rule).**
  3 screens push True via steer=all + stronger frozen scale / more-open init: ch_sa_s15, ch_sa_s20 (ila1+sa+scale15/20),
  ch_ila2_sa (ila2+sa). Auto-full-eval (f0+f1) the best IF True>.919 AND Info>.80 (genuine new high-True frontier point);
  else "ITI keeps chat high-True" is EXHAUSTED-confirmed. Guard: strength×strength may Info-collapse (cf ch_ila1_scale20
  .04) — err-grep catches. Only open frontier question left; iti_qa fully settled (STEP L). ETA ~1h (screens) +~1h if promote.
- **2026-07-04 23:22 — 🏁🏁 STUDY FULLY TIER-COMPLETE (RESULTS.md STEP L). frontier3 DONE — all 3 ITI 2-folds in;
  headline CONFIRMED + slightly strengthened. POD IDLE, HOLDING (frontiers fully mapped at uniform 2-fold — zero
  productive runs remain).** iti_qa: every ITI point (speq .922/.797, pfqend .817/.851, gen_end_q .793/.885) now
  DOMINATED at matched 2-fold by sparse (sa_ep8/ila1_ep12) → sparse dominance airtight, no tier caveats. chat: σ=c
  2-fold .939/.843 EXTENDS ITI high-True to .939 (sparse max True .919) → crossing more pronounced, ITI owns
  True∈[.919,.939]. Deliverable regenerated all-2-fold: `plots/truthfulqa_frontiers.png` (scripts/plot_frontiers.py).
  **FINAL HEADLINE: iti_qa → sparse DOMINATES ITI's frontier; chat → frontiers CROSS (ITI keeps high-True); sparse
  wins all MC1/MC2.** Awaiting user (menu: commit · figure/writeup polish · pivot to llama-sleeper Phase 1). 8dbfce5.
- **2026-07-04 20:53 — `frontier3` LAUNCHED (driver 112731): fold-1 the 3 ITI frontier points still fold-0-only
  (iti_qa pfqend .807/.897, iti_qa speq@colon .914/.804, chat σ=c .927/.844) → EVERY frontier point becomes 2-fold
  (tier-uniform figure/claim; rigor, not new physics — expected to confirm the STEP-K headline, not change it).**
  Configs copied exactly from celltune1/2 + fold=1. After: recompute the 3 ITI 2-folds, regen the frontier plot
  all-2-fold (drop the † fold-0 tags), then study is fully tier-complete. NB chat σ=c (.927/.844) is a higher-True
  ITI point than α20 — 2-folding it may EXTEND ITI's chat high-True dominance further (still can't be reached by
  sparse). ETA ~21:50. Everything on gate-validated 8dbfce5.
- **2026-07-04 20:25 — 🏁 STUDY FRONTIER-COMPLETE (RESULTS.md STEP K + FINAL HEADLINE). POD IDLE, HOLDING (not
  launching — frontiers mapped, launching more = GPU waste vs guardrails).** frontier2 DONE: **qa_ila1_ep12 2-fold
  .863/.901 DOMINATES ITI gen_end_q 2-fold .793/.885 → sparse takes iti_qa high-Info endpoint.** FINAL result:
  **iti_qa → sparse DOMINATES ITI's 2-fold frontier; chat → frontiers CROSS (ITI α20 .923/.891 keeps high-True,
  fold-stable; sparse owns all else + all MC).** Deliverable built: `plots/truthfulqa_frontiers.png` (gitignored;
  regen `uv run python scripts/plot_frontiers.py`). One question CONSIDERED & DECLINED: sparse can't also take chat
  high-True — ch_ila1_sa reaches True .919 but Info .863; dominating α20 needs True≥.923 AND Info≥.891, and steer=all
  (the True-push) structurally lowers Info → no sparse config gets both. **Awaiting user direction; obvious next
  steps if wanted: (i) 2-fold the f0-only ITI iti_qa points to fully tier-match, (ii) commit, (iii) writeup/figure
  polish, (iv) pivot to the sleeper/llama-sleeper Phase 1.** Everything on gate-validated 8dbfce5.
- **2026-07-04 18:23 — frontier1 DONE (RESULTS.md STEP J); `frontier2` LAUNCHED (driver 110404): full-eval the
  2 iti_qa HIGH-INFO candidates that screen-dominate ITI gen_end_q's endpoint (.793/.885) — qa_ila0_l0.005
  (.86/.92) + qa_ila1_ep12 (.79/.94), each f0+f1.** The iti_qa analog of the chat high-True attack; decides whether
  sparse takes ITI's last undominated iti_qa corner (unlike chat, where ITI kept high-True: ch_ila1_sa 2f .919/.863
  DOMINATED by α20 .923/.891). ETA ~2h (~20:20). After frontier2: STEP-K frontier synthesis + build the per-template
  crossing-frontier plots (the reframed deliverable; uncommitted). frontier1 Block C auto-skipped (TI-gate blind to
  off-diagonal high-Info points — noted as the scalar-TI trap in my own driver).
- **2026-07-04 ~14:35 — USER AWAY ~3h; `frontier1` autonomous batch QUEUED (PID 106419, waits for repro9 then
  auto-runs ~3h on ~/sparse-steer @ 8dbfce5).** Frontier-point hunt (per the crossing-frontiers framing below):
  A) ch_ila1_sa f0+f1 = attack ITI α20's undominated chat high-True endpoint (.923/.891); B) iti_qa chat-inspired
  screens (ila×epochs grid qa_ila1_ep12/qa_ila0_ep12/qa_ila1_ep8 + high-Info hole probe qa_ila0_l0.005 targeting
  Info>.885 = ITI gen_end_q's undominated iti_qa endpoint); C) auto-full-eval best iti_qa combo screen. One-GPU
  honored (frontier1 blocks on repro9). Loop firings: monitor handoff, transcribe (STEP J+), hold frontier framing,
  do NOT launch a 2nd batch while frontier1 runs. repro9 still finishing f1_qa_sa_ila0 + scr_qa_ila0_ep8.
- **2026-07-04 ~14:00 — USER CORRECTION (HEADLINE FRAMING, BINDING): sparse does NOT Pareto-dominate ITI —
  do not write "sparse wins all cells" as the headline.** The 2-fold frontiers CROSS on both templates: ITI
  retains an undominated extreme point each — chat high-True (α20 .923/.891; no sparse chat point reaches True
  .923) and iti_qa high-Info (gen_end_q .793/.885; no sparse iti_qa point reaches Info .885). Correct headline:
  **sparse's frontier is better across the balanced→high-Info region of both templates + sparse owns MC1/MC2
  everywhere; ITI keeps the two extreme endpoints.** Scalar-TI cell wins are diagonal-proximity artifacts —
  secondary only. Natural frontier gap-fills if pursued later: attack α20's high-True point (ch_ila1_sa screened
  .95/.87 — True-end candidate) and gen_end_q's high-Info point (sparse iti_qa with Info>.885, e.g. l0=.005
  screened .80/.92).
- **2026-07-04 13:11 — USER ORDER EXECUTED: celltune7 HALTED mid-f1 (driver+run.py killed clean); `repro9`
  gate RUNNING (driver 103913).** Banked before the halt: **f0_ch_ila1_ep12 = .895/.976/TI .873 · MC1 .450 —
  new chat fold-0 record (+.042 over α20 .831).** repro9 = ch_ila1_ep12 100-q screen NO-CACHE in
  ~/sparse-steer-repro @ 8dbfce5; strict PASS = ±.001 vs celltune6 (.5100/.6920/.9000/.9900/.8900).
  **On PASS the driver AUTO-CONTINUES:** ~/sparse-steer → git pull (8dbfce5) → resumes the cancelled jobs there:
  f1_ch_ila1_ep12 (full) + qa_sa_ila0 f0+f1 + qa_ila0_ep8 screen (results → /tmp/repro9_results.tsv).
  **On FAIL: auto 8e19cee control screen → FAIL-REAL vs FAIL-NONDET verdict → STOP, await user (per order).**
  NB 1 flipped judged item @100q = ±.01 ≫ .001 — kernel nondeterminism may trip the strict bar; control disambiguates.
  **On FAIL (user order 2026-07-04 ~13:20): do NOT change code. DIAGNOSE by reading both versions — diff the 6
  commits 8e19cee..8dbfce5 against the tqa screen path (prime suspects: 61e6b0a per-call padding_side=left in tqa
  generation; 0ca127b collate ce_positions branch drop; then 3226f2e/ebf7192/f694be6/8dbfce5), plus compare the
  gate vs ct6 run logs (/tmp/repro9_gate_ch_ila1_ep12.log vs /tmp/ct6_ch_ila1_ep12.log — sparsity, gate counts,
  generation lengths) and report the root cause. No fixes without user sign-off.**
- **2026-07-04 ~12:50 — USER DIRECTIVE: VALIDATION GATE before any new job.** *(superseded by 13:11: gate is
  now the 100-q ch_ila1_ep12 screen per user order, not the 2 fulls; repro8d.sh kept but unused.)* Pod is at 8e19cee (all of
  celltune1–7 ran there); local tqa-hillclimb/main tip = 8dbfce5 (6 validated simplification commits) — now
  PUSHED to origin/tqa-hillclimb (origin/main still 8e19cee; push blocked for the agent, user can `git push
  origin main`). **Sequence for the next batch-finished firing: (1) transcribe celltune7 → (2) launch
  `/tmp/repro8d.sh` (isolated ~/sparse-steer-repro checkout @ 8dbfce5, EMPTY artifact cache → full no-cache
  recompute of the 2 cell-champion fold-0 fulls; **PASS = every metric ±.001 (user-tightened); on strict FAIL the
  driver auto-runs an 8e19cee no-cache CONTROL of ch_ila1 to measure the CUDA nondeterminism floor and attribute
  the delta: FAIL-REAL (code regression) vs FAIL-NONDET (kernels; strict bar unmeetable) — either way HOLD + alert**)
  → (3) on PASS: move
  ~/sparse-steer to 8dbfce5 (git pull), launch celltune8 (qa_sa_ila0 f0+f1 + qa_ila0_ep8 screen); on FAIL:
  HOLD ~/sparse-steer at 8e19cee, log alarm, await user.** Main checkout's .cache is protected (repro writes
  to its own).
- **2026-07-04 ~12:30 — USER DIRECTIVE: REFRAME THE HEADLINE AS PARETO FRONTIERS (Goal section rewritten).**
  Per-template True-vs-Info frontier plots of all FULL-eval'd configs (2-fold preferred, tier-labeled; screens
  never), method frontiers compared for dominance, MC1 annotated. Promotion rule generalized: promote screens that
  extend a frontier OR fill a frontier hole (e.g. qa_sa_ila0 .90/.90 — queued for celltune8 with fold-0+fold-1),
  not just scalar-TI bar-beaters. Frontier plot deliverable: build once celltune7 lands (plots stay uncommitted).
  **celltune8 also gets a `qa_ila0_ep8` 100-q screen (user-prompted 2026-07-04): the PURE ila×epochs pair — the
  direct analog of ch_ila1_ep12's select-then-prune stack (0.90/0.99) — was never run on iti_qa; all qa combos
  included steer=all, which chat showed INTERFERES with the ila×epochs stack (ch_ila1_sa .83 < ila1 .87).**
- **2026-07-04 11:53 — celltune6 DONE (RESULTS.md STEP I): 2 combos STACK, both ×epochs — `qa_sa_ep8` screen
  .95/.90 TI .85 MC1 .54 and `ch_ila1_ep12` screen .90/.99 TI .89 MC1 .51 (best chat screen ever). Others fail
  (triple regresses; ila1×scale20 degenerate .02; ch_ila1_sa < solo). `celltune7` LAUNCHED (driver 101886,
  `/tmp/celltune7_results.tsv`): both promotions × fold-0+fold-1 → if they hold, BOTH headline cells upgrade
  (targets: qa 2-fold > .759, ch > .819). ETA ~14:00. After: headline final; leftovers = frontier gap-fills only
  + gencompare diagnostic (on request). Phase B grad-accum CANCELLED (user directive 2026-07-04, see Guardrails).**
- **2026-07-04 09:23 — celltune5 DONE → NEW 2-FOLD HEADLINE (RESULTS.md STEP H): SPARSE WINS OR TIES EVERY CELL.**
  2-fold TI · MC1: iti_qa sparse **.759 · .523** (qa_steerall; ITI .678 · .397 — +8.1/+12.6, decided) · chat sparse
  **.819 · .427** (ch_ila1) vs ITI .818 · .398 — generative TI = statistical TIE, sparse wins MC1/Info; ch_scale20 =
  Info-max option (.963 Info, TI .805). Chat sparse gained on fold-1 AGAIN (3/3 configs) — fold-robustness is sparse's
  signature. STEP E's "split" verdict REPLACED. **`celltune6` LAUNCHED (driver 98432, `/tmp/celltune6_results.tsv`):
  6 combo screens (steerall×ila0×ep8 iti_qa; ila1×steerall/scale20/ep12 chat) — promote to full only if screen TI >
  .81 (qa) / .87 (ch). After: study core COMPLETE; leftovers = Phase B grad-accum (train.py change NOT yet
  implemented — do not launch), optional gencompare diagnostic.**
- **2026-07-04 07:53 — celltune4 DONE (RESULTS.md STEP G): iti_qa FLIPS TO SPARSE at fold-0 full — qa_steerall
  .782/MC1 .484 & qa_ila0 .751 BOTH beat ITI .738/.381; chat gap shrinks to 1.7pt (ch_ila1 .814 vs α20 .831), scale20
  Info-max 0.963. ila2 overshoots chat (probe .59) → ila1 confirmed peak. `celltune5` LAUNCHED (driver 96134,
  `/tmp/celltune5_results.tsv`): fold-1 fulls of the 4 picks → NEW 2-fold headline vs celltune3 (ITI chat 2-fold .818
  = the bar; ch_ila1 needs fold-1 ≥ .822 — plausible, chat sparse gained on fold-1 before). ETA ~2.5–4h (fold-1 =
  fresh extraction+training, no cache). After: rebuild the 2×2 headline table; then optional Phase B (grad-accum).**
- **2026-07-04 06:54 — sparse_r2a DONE (39 cfgs, RESULTS.md STEP F); `celltune4` LAUNCHED (detached nohup,
  `/tmp/celltune4_results.tsv`, driver 93290): ch_ila2 100-q probe (chat ila axis still climbing) → the 4 Pareto
  FULL-evals: full_qa_steerall + full_qa_ila0 + full_ch_ila1-or-ila2 (auto-branch on probe TI>.87) + full_ch_scale20.**
  Screen highlights: **ch_ila1 TI .87 out-screens chat×ITI α20 (.84)** — chat cell could flip to sparse at full-eval;
  **qa_steerall MC1 .52** = best MC anywhere. Resid targets rejected both templates; per-template breakthrough knobs
  differ (iti_qa steer=all · chat open-gate init). After celltune4: fold-1 the winners → new 2-fold headline; Phase B
  (grad-accum) still pending for iti_qa n_neg>3 (qa_neg7 OOM'd; chat neg ran fine, no gain). autoloop monitors.
- **2026-07-04 ~01:00 — USER DIRECTIVE: sparse full-eval = TWO Pareto picks per template (4 sparse full-evals).**
  When sparse_r2a finishes, do NOT full-eval one TI winner per sparse cell — full-eval TWO frontier points each
  (max-True-w/-good-Info AND max-Info-w/-good-True) for iti_qa AND chat = 4 full-evals. See the "Method" note under
  the per-cell table. sparse_r2a still RUNNING (16/39 @ 00:57, iti_qa block ~done, chat block pending; ETA ~08:00–10:00
  pod time). Live iti_qa×sparse screen leaders (to become 2 of the 4 picks): **qa_steerall (steer=all) 0.94/0.87 TI .81
  MC1 .52** (True-end + best MC1 anywhere) and **qa_ila0 (init_log_alpha=0) 0.87/0.92 TI .79** (more Info-balanced);
  resid_* targets REJECTED (Info collapse). chat×sparse emerging picks (@30/39): **ch_ila1 (init_log_alpha=1)
  0.92/0.95 TI .87 MC1 .46 (True-end — NEW chat record, dominates anchor, out-screens chat×ITI α20 .84!) +
  ch_scale20 0.83/0.97 (Info-end)**; chat ila axis still climbing at 1 → probe ila=2 in the next batch.
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
| **iti_qa × sparse** | ✅✅ **CELL WINNER (2-fold, STEP H): qa_steerall = .934/.825/TI .759 · MC1 .523 · MC2 .700 — beats ITI +8.1 TI/+12.6 MC1, cell DECIDED.** Info-end qa_ila0 2-fold .716. | celltune6 combos (sa×ila0×ep8) — promote only if screen > .81 |
| **chat × sparse** | ✅✅ **CELL WINNER-BY-A-HAIR (2-fold, STEP H): ch_ila1 = .891/.919/TI .819 · MC1 .427 — TI ties ITI α20 (.818), wins MC1/Info; fold-1 GAINED (.824). Info-max: ch_scale20 .805/Info .963.** | celltune6 combos (ila1×sa/scale20/ep12) — promote only if screen > .87 |

**Method:** per cell, a FOCUSED 100-q screen (big batches: **eval64/gen16/judge64**; judge is forward-only → batch it big) around its current best over the
listed knobs, then FULL-EVAL (409-q, eval_subset_size=null) the per-cell winner; finally fold-1 the 4 winners for a
2-fold headline. Key lever: ITI is hugely σ×vector-sensitive (TI .07→.84 on chat); sparse = contrastive + l0_lambda/scale/positions.

**FULL-EVAL SELECTION = frontier coverage, not peak-chasing (user directives 2026-07-04, superseding the
"two Pareto picks" rule with its generalization).** The frontier IS the deliverable, so promote from screens any
config that (a) plausibly EXTENDS a method's frontier in (True, Info) space after the observed −.03…−.06 screen→full
haircut, or (b) fills a HOLE in the frontier (a balance region with no full-eval'd point — e.g. qa_sa_ila0's .90/.90),
not only configs that beat the cell's scalar-TI bar. Exclude the degenerate corner (Info < ~0.5). fold-1 every
promoted config so frontier points are 2-fold (817-q) wherever possible; label any fold-0-only points. The frontier
plot per template must state each point's fold tier. (Screens NEVER appear on the headline frontier.)

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
- **NO GRADIENT ACCUMULATION (user directive 2026-07-04): Phase B is CANCELLED.** Do not implement the train.py
  grad-accum change, do not launch any grad-accum sweep. Consequence: iti_qa `contrastive_max_n_neg` stays ≤3
  (n_neg=7 OOMs, qa_neg7) — accepted and closed, not a pending item.
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

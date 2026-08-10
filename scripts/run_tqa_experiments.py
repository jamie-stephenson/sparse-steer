"""Run the full TruthfulQA study (v2 protocol), chaining sparse_steer directly.

The config-keyed artifact cache is the single source of truth. Workers only compute:
each job composes its config, runs build_experiment().run(), and the experiment code
caches steering artifacts and eval metrics automatically. The parent decides what still
needs to run by looking each job's eval result up in that cache (no model load), and
regenerates fulls.tsv / caps.tsv afterwards as harvest views of the cache -- the TSVs
are never load-bearing, and a failed job simply never caches, so it retries on the next
invocation. caps.tsv keeps the "KEY: value" metrics format that report/diss/scripts/
make_figures.py parses.

Per cell (model x template), two factorial grids, each config run on both CV folds:
  Sparse:  l0_lambda {0, 0.005, 0.01} x init_log_alpha {def:-0.79, open:1}
           x steer position {all, answer_gen}                (num_epochs=16 fixed)
  ITI:     scale {8, 15, 22} x topk {24, 48, 96}
           x steer position {all, answer_gen}    (sigma=gen_end_q, or --iti-sigma)

Stages (--stages, comma list, default all):
  anchors  unsteered 2-fold full evals (calibration)
  grid     2-fold full True/Info (+MC) on every grid config
  promote  2-fold means + per-(cell,method) Pareto frontier     -> promoted.tsv
           (also harvests gate_density.tsv from the cached sparse artifacts)
  caps     capability suite on the frontier only, the v2 protocol's variants:
           loglik MMLU/ARC/wikitext-CE (fixed leaderboard template, 0-shot) +
           generative MMLU/ARC under BOTH the fixed and the chat template.
           Chat-template loglik was dropped from the protocol (lm-eval wraps the
           completion-style primer in a chat turn and scores a bare letter as the
           assistant reply, which went ~random at 0-shot); chat capability is
           measured generatively instead.

Multi-GPU: with N visible GPUs (or --ngpu N) the parent spawns one worker process per
GPU, all pulling from a shared job queue -- dynamic load balancing, with jobs
interleaved across cells so concurrent workers hold different models and never
duplicate a shared extraction artifact. Promotion is a barrier between grid and caps.

  uv run python scripts/run_tqa_experiments.py                       # everything, all GPUs
  uv run python scripts/run_tqa_experiments.py --ngpu 2 --stages grid --cells ll_qa,qw_qa
  uv run python scripts/run_tqa_experiments.py --list                # plan + cache status
"""
import argparse
import gc
import os
import re
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str(ROOT / "configs")
SCRATCH = ROOT / "scripts" / "scratch"

CELLS = {
    "ll_qa": ["task=truthfulqa"],
    "ll_ch": ["task=truthfulqa", "prompt_template=chat", "extraction_template=chat"],
    "qw_qa": ["task=truthfulqa_qwen"],
    "qw_ch": ["task=truthfulqa_qwen", "prompt_template=chat", "extraction_template=chat"],
    "base_qa": ["task=truthfulqa", "model_name=huggyllama/llama-7b", "++model_dtype=float16"],
}

# eval/gen/judge batch sizes per chip tier: 80GB chips (A100/H100/H200) take the
# large tier, anything else (A40 pods, CPU-only hosts) keeps the original sizes.
# These sizes are part of the eval-artifact cache keys, so each tier addresses its
# own eval artifacts; extraction and gate-training artifacts are keyed by
# extract/train batch sizes instead and are shared across tiers.
BATCH_TIERS = {
    "large": {"ll": (160, 40, 80), "qw": (64, 16, 32)},
    "a40": {"ll": (64, 16, 32), "qw": (32, 8, 16)},
}
CELL_FAMILY = {"ll_qa": "ll", "ll_ch": "ll", "qw_qa": "qw", "qw_ch": "qw", "base_qa": "ll"}

_TIER: str | None = None


def batch_overrides(cell) -> list[str]:
    """Chip-conditional eval/gen/judge batch sizes for one cell. The tier is detected
    lazily on first use, never at module import: spawn workers import this module
    before pinning CUDA_VISIBLE_DEVICES, and importing torch that early would break
    the pin."""
    global _TIER
    if _TIER is None:
        try:
            import torch

            name = torch.cuda.get_device_name(0)
        except Exception:
            name = ""
        _TIER = "large" if re.search(r"A100|H100|H200", name) else "a40"
    e, g, j = BATCH_TIERS[_TIER][CELL_FAMILY[cell]]
    return [f"eval_batch_size={e}", f"gen_batch_size={g}", f"judge_batch_size={j}"]

COMMON = ["disjoint_extract_refine_data=false", "extraction_mcq_mode=mc2"]
SPARSE = ["method=sparse", "train_batch_size=1", "+contrastive_weight=1", "+ce_weight=0",
          "track_gates=false", "extract_token_position=completion_final",
          "+contrastive_max_n_neg=3", "init_raw_scale=15", "num_epochs=16"]
ITI = ["method=iti", "extract_token_position=completion_final", "iti_probe_device=cuda"]

SP_L0 = ["0", "0.005", "0.01"]
SP_ILA = [("def", "-0.79"), ("open", "1")]
POS = [("all", "all"), ("ag", "answer_gen")]
ITI_A = ["8", "15", "22"]
ITI_K = ["24", "48", "96"]

FULLS_HDR = "tag\tcell\tmethod\tfold\ttrue\tinfo\tmc0\tmc1\tmc2\targs\n"
CAPS_HDR = "tag\tcell\tmethod\tstage\tmetrics\n"

# capability suite on the promoted frontier: each cell gets exactly FIVE evals, in the
# cell's own template regime -- plain cells (iti_qa-derived steering) run leaderboard-style
# loglik + generative, chat cells run chat-template loglik + generative. Per benchmark one
# loglik and one generative job (mmlu, arc_challenge) plus one wikitext perplexity job.
# Steering positions: mmlu/arc follow the promoted config's own steer_token_position
# (the regime the method was trained/evalled in); wikitext steers ALL positions.
_LL_MMLU = ["lmeval_tasks=[mmlu]", "lmeval_limit=100", "lmeval_fewshot=0"]
_LL_ARC = ["lmeval_tasks=[arc_challenge]", "lmeval_fewshot=0"]
_LL_WIKI = ["lmeval_tasks=[wikitext]", "lmeval_steer=all"]
_CT_LL = ["lmeval_chat_template=true", "lmeval_fewshot_multiturn=true"]
_GEN = ["inspect_eval_limit=1000", "inspect_max_tokens=64"]


def _cfg_steer_pos(cfg: list[str]) -> str:
    """The promoted config's own steer position ("all" for unsteered rows, where no
    steering fires and the value only keys the artifact)."""
    for a in cfg:
        if a.startswith("steer_token_position="):
            return a.split("=", 1)[1]
    return "all"


def grid_jobs(cell, iti_sigma="gen_end_q"):
    """(tag, method, config overrides) for one cell's sparse + ITI grids."""
    for l0 in SP_L0:
        for ilab, ila in SP_ILA:
            for plab, pos in POS:
                yield (f"sp_{cell}_l{l0}_{ilab}_{plab}", "sparse",
                       SPARSE + ["generative_eval=true", f"l0_lambda={l0}",
                                 f"gate_config.init_log_alpha={ila}",
                                 f"steer_token_position={pos}"])
    for a in ITI_A:
        for k in ITI_K:
            for plab, pos in POS:
                yield (f"iti_{cell}_a{a}_k{k}_{plab}", "iti",
                       ITI + [f"iti_sigma_position={iti_sigma}", "generative_eval=true",
                              f"iti_scale={a}", f"iti_topk={k}",
                              f"steer_token_position={pos}"])


def full_jobs(cell, stages, iti_sigma="gen_end_q"):
    """("full", tag, cell, method, fold, cfg) for one cell's anchors + grid."""
    if "anchors" in stages:
        for fold in (0, 1):
            yield ("full", f"uns_{cell}", cell, "unsteered", fold, ["method=unsteered"])
    if "grid" in stages:
        for tag, method, cfg in grid_jobs(cell, iti_sigma):
            for fold in (0, 1):
                yield ("full", tag, cell, method, fold, cfg)


def full_overrides(args, cell, fold, cfg) -> list[str]:
    return (COMMON + [f"device={args.device}"] + CELLS[cell] + batch_overrides(cell)
            + ["eval_subset_size=null", f"fold={fold}"] + cfg)


def cap_overrides(args, cell, cfg) -> list[str]:
    return (COMMON + [f"device={args.device}"] + CELLS[cell] + batch_overrides(cell)
            + ["eval_subset_size=2", "generative_eval=false"] + cfg)


def job_overrides(args, job) -> list[str]:
    if job[0] == "full":
        _, _tag, cell, _method, fold, cfg = job
        return full_overrides(args, cell, fold, cfg)
    _, _tag, cell, _method, _stage, cfg = job
    return cap_overrides(args, cell, cfg)


# ── config composition + cache access (the cache IS the resume state) ────────

def compose_experiment(overrides: list[str]):
    from hydra import compose
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict
    from sparse_steer.experiment import build_experiment
    from sparse_steer.tasks.truthfulqa.task import TruthfulQATask

    cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)
    return build_experiment(cfg, TruthfulQATask())


def cached_metrics(overrides: list[str]) -> dict | None:
    """Read a job's eval metrics straight from the artifact cache, without loading a
    model (Experiment.__init__ is config-only). None = not run yet / failed / uncached."""
    from sparse_steer.utils.cache import load_cached_json

    exp = compose_experiment(overrides)
    hit = exp._try_cache_lookup(exp._eval_artifact_type)
    return load_cached_json(hit.artifact_path) if hit else None


def run_job(args, desc: str, overrides: list[str], label: str = ""):
    """Compute one config; all results land in the cache via the experiment code."""
    import torch
    from sparse_steer.utils.memory import free_model_memory

    print(f"[{time.strftime('%H:%M')}{label}] {desc}", flush=True)
    exp = None
    try:
        exp = compose_experiment(overrides)
        exp.run()
    except Exception as e:  # no cache entry is written, so the job retries next run
        print(f"ERR {desc}: {type(e).__name__}: {e}", flush=True)
    del exp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_model_memory()


# ── execution: serial, or one worker process per GPU off a shared queue ──────

def interleave(lists):
    """Round-robin merge, so adjacent jobs (= concurrent workers) come from
    different cells and therefore hold different models."""
    return [j for group in zip_longest(*lists) for j in group if j is not None]


def gpu_list(args) -> list[str]:
    if args.device != "cuda":
        return []
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        ids = [x for x in env.split(",") if x]
    else:
        try:
            import torch
            ids = [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            ids = []
    return ids[: args.ngpu] if args.ngpu else ids


def job_desc(job) -> str:
    return f"FULL {job[1]} fold={job[4]}" if job[0] == "full" else f"CAP {job[1]}"


def _worker(args, gpu: str, queue):
    # pin the GPU and give this worker its own inspect log dir BEFORE torch is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["INSPECT_LOG_DIR"] = f"logs/gpu{gpu}"
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        while (job := queue.get()) is not None:
            run_job(args, job_desc(job), job_overrides(args, job), label=f" gpu{gpu}")


def execute(args, jobs):
    """One GPU (or cpu) runs serially inside the parent's hydra context; several GPUs
    run spawn workers off a shared queue. Workers never write shared files."""
    if not jobs:
        return
    gpus = gpu_list(args)
    if len(gpus) <= 1:
        for job in jobs:
            run_job(args, job_desc(job), job_overrides(args, job))
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")  # fresh interpreters: no inherited CUDA/hydra state
    queue = ctx.Queue()
    for job in jobs:
        queue.put(job)
    for _ in gpus:
        queue.put(None)
    workers = {g: ctx.Process(target=_worker, args=(args, g, queue)) for g in gpus}
    for w in workers.values():
        w.start()
    while workers:
        for g, w in list(workers.items()):
            w.join(timeout=10)
            if w.exitcode is None:
                continue
            del workers[g]
            if w.exitcode != 0 and not queue.empty():
                # a device-side CUDA assert poisons the worker's context and kills the
                # process mid-job; its shutdown sentinel is still queued, so a fresh
                # replacement on the same GPU can drain the remaining jobs
                print(f"WARNING: worker gpu{g} died (exit {w.exitcode}); respawning", flush=True)
                workers[g] = ctx.Process(target=_worker, args=(args, g, queue))
                workers[g].start()
            elif w.exitcode != 0:
                print(f"WARNING: worker gpu{g} exited nonzero near shutdown", flush=True)


# ── harvest: regenerate the TSVs as views of the cache ───────────────────────

def harvest_fulls(args) -> Path:
    fulls = Path(args.results_dir) / "fulls.tsv"
    fulls.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in CELLS:  # always all cells: the TSV is the full current view
        for job in full_jobs(cell, {"anchors", "grid"}, args.iti_sigma):
            _, tag, cell_, method, fold, cfg = job
            m = cached_metrics(full_overrides(args, cell_, fold, cfg))
            if m is None:
                continue

            def fmt(key):
                v = m.get(key)
                return f"{v:.4f}" if isinstance(v, (int, float)) else ""

            rows.append("\t".join(
                [tag, cell_, method, str(fold), fmt("gen_truthful"), fmt("gen_informative"),
                 fmt("mc0"), fmt("mc1"), fmt("mc2"), " ".join(cfg)]))
    fulls.write_text(FULLS_HDR + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} cached rows -> {fulls}", flush=True)
    return fulls


def promote(args):
    """2-fold means + Pareto promotion over the harvested fulls.tsv (scratch transforms)."""
    res = Path(args.results_dir)
    g2f = res / "grid_2fold.tsv"
    subprocess.run([sys.executable, str(SCRATCH / "sweep_fold_mean.py"),
                    str(res / "fulls.tsv"), str(g2f)], check=True)
    subprocess.run([sys.executable, str(SCRATCH / "sweep_promote.py"), str(g2f),
                    "--cap", str(args.promote_cap), "--out", str(res / "promoted.tsv")],
                   check=True)
    print((res / "promoted.tsv").read_text())


def harvest_gate_density(args):
    """Learned gate density per cached sparse steering artifact of the grid, both folds ->
    gate_density.tsv. Per config: the eval-mode L0 -- a gate is active iff, with the
    deterministic hard-concrete gate z = clamp(sigmoid(log_alpha)*(high-low)+low, 0, 1)
    under the artifact's own gate_config, z >= eval_threshold (the exact SteeringHook eval
    branch). Columns are a superset of the file report/diss/scripts/make_figures.py reads
    (it keys on model/prompt_template/l0_lambda/init_log_alpha/steer_pos and reads
    density); tag/fold are prepended. Reads the cached tensors only, no model load;
    configs whose artifact is not cached yet are skipped."""
    import torch
    from sparse_steer.utils.cache import ArtifactType, lookup as cache_lookup

    rows = []
    for cell in CELLS:
        for tag, method, cfg in grid_jobs(cell, args.iti_sigma):
            if method != "sparse":
                continue
            for fold in (0, 1):
                exp = compose_experiment(full_overrides(args, cell, fold, cfg))
                hit = cache_lookup(ArtifactType.SPARSE_STEERING, exp.config,
                                   exp.task.task_name,
                                   **exp._cache_kwargs(ArtifactType.SPARSE_STEERING))
                if hit is None:
                    continue
                try:
                    d = torch.load(hit.artifact_path, map_location="cpu",
                                   weights_only=False)
                except Exception as e:
                    print(f"gate_density: skipping {tag} fold={fold}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                low, high = d["gate_config"]["stretch_limits"]
                thr = d["gate_config"]["eval_threshold"]
                sd = d["state_dict"]
                n_active = n_total = 0
                active_sites = []  # (layer, component, head_or_-1)
                per_comp = {}      # component -> [active, total]
                for k in sorted(k for k in sd if k.endswith("log_alpha")):
                    z = torch.clamp(
                        torch.sigmoid(sd[k].float()) * (high - low) + low, 0.0, 1.0)
                    act = z >= thr
                    n_total += z.numel()
                    n_active += int(act.sum())
                    m = re.match(r"hooks\.(.+)_(\d+)\.log_alpha", k)
                    comp, layer = m.group(1), int(m.group(2))
                    pc = per_comp.setdefault(comp, [0, 0])
                    pc[0] += int(act.sum())
                    pc[1] += z.numel()
                    for j in act.nonzero(as_tuple=True)[0].tolist():
                        active_sites.append((layer, comp, j if z.numel() > 1 else -1))
                cf = hit.manifest.config_fields
                rows.append({
                    "tag": tag,
                    "fold": fold,
                    "model": cf.get("model_name", "?").split("/")[-1],
                    "prompt_template": cf.get("prompt_template"),
                    "l0_lambda": (cf.get("l0_lambda")
                                  if cf.get("l0_lambda") is not None else 0.0),
                    "init_log_alpha": (cf.get("gate_config") or {}).get("init_log_alpha"),
                    "steer_pos": cf.get("steer_token_position"),
                    "targets": ",".join(cf.get("targets") or []),
                    "source_hash": hit.manifest.source_hash[:8],
                    "n_active": n_active,
                    "n_total": n_total,
                    "density": round(n_active / n_total, 4) if n_total else 0,
                    "per_component": ",".join(
                        f"{c}:{a}/{t}" for c, (a, t) in sorted(per_comp.items())),
                    "active_sites": ";".join(
                        f"{l}/{c}" + (f"/h{h}" if h >= 0 else "")
                        for l, c, h in active_sites[:60]),
                })
    cols = ["tag", "fold", "model", "prompt_template", "l0_lambda", "init_log_alpha",
            "steer_pos", "targets", "source_hash", "n_active", "n_total", "density",
            "per_component", "active_sites"]
    out = Path(args.results_dir) / "gate_density.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\t".join(cols) + "\n"
                   + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r in rows))
    print(f"harvested gate density for {len(rows)} cached sparse artifacts -> {out}",
          flush=True)


def cap_points(args, cell):
    """(tag, method, config overrides) to cap: unsteered + the promoted frontier.
    Column positions come from the header so promote-side format changes cannot
    silently drop the frontier (a fixed args-at-column-9 assumption once skipped
    every frontier row when promote switched to a 4-column TSV)."""
    yield "uns", "unsteered", ["method=unsteered"]
    promoted = Path(args.results_dir) / "promoted.tsv"
    if not promoted.exists():
        return
    lines = promoted.read_text().splitlines()
    if not lines:
        return
    header = lines[0].split("\t")
    try:
        i_tag, i_cell = header.index("tag"), header.index("cell")
        i_method, i_args = header.index("method"), header.index("args")
    except ValueError as e:
        raise RuntimeError(f"promoted.tsv header {header!r} missing a required column") from e
    n_yielded = 0
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) > i_args and f[i_cell] == cell:
            n_yielded += 1
            yield f[i_tag], f[i_method], f[i_args].split()
    if n_yielded == 0:
        print(f"  WARNING: promoted.tsv yielded no frontier rows for cell {cell}", flush=True)


def cap_jobs(args, cell):
    """("cap", tag, cell, method, stage, overrides) tuples for one cell: five evals per
    promoted point, in the cell's native template regime (chat cells: chat-template loglik
    + chat-template generative; plain cells incl. base: leaderboard loglik + fixed-format
    generative)."""
    chat = cell.endswith("_ch")
    for ptag, method, cfg in cap_points(args, cell):
        pos = _cfg_steer_pos(cfg)
        ll_extra = [f"lmeval_steer={pos}"] + (_CT_LL if chat else [])
        gen_extra = [f"inspect_steer={pos}",
                     "inspect_apply_template=true" if chat else "inspect_apply_template=false"]
        regime = "ct" if chat else "fx"
        yield ("cap", f"cap_mmll_{cell}_{ptag}", cell, method, f"loglik-{regime}-mmlu",
               cfg + _LL_MMLU + ll_extra)
        yield ("cap", f"cap_arcll_{cell}_{ptag}", cell, method, f"loglik-{regime}-arc",
               cfg + _LL_ARC + ll_extra)
        yield ("cap", f"cap_mmgen_{cell}_{ptag}", cell, method, f"generative-{regime}-mmlu",
               cfg + ["inspect_evals=[mmlu]"] + _GEN + gen_extra)
        yield ("cap", f"cap_arcgen_{cell}_{ptag}", cell, method, f"generative-{regime}-arc",
               cfg + ["inspect_evals=[arc_challenge]"] + _GEN + gen_extra)
        yield ("cap", f"cap_wiki_{cell}_{ptag}", cell, method, "loglik-wikitext",
               cfg + _LL_WIKI)


def harvest_caps(args):
    caps = Path(args.results_dir) / "caps.tsv"
    rows = []
    for cell in CELLS:
        for job in cap_jobs(args, cell):
            _, tag, cell_, method, stage, cfg = job
            m = cached_metrics(cap_overrides(args, cell_, cfg))
            if m is None:
                continue
            # the "KEY: value" format report/diss/scripts/make_figures.py::load_caps parses
            keep = ("MMLU", "ARC", "WIKITEXT")
            metrics = " ".join(f"{k.upper()}: {v:.4f}" for k, v in sorted(m.items())
                               if isinstance(v, (int, float)) and k.upper().startswith(keep))
            rows.append(f"{tag}\t{cell_}\t{method}\t{stage}\t{metrics}")
    caps.write_text(CAPS_HDR + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} cached cap rows -> {caps}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="anchors,grid,promote,caps")
    p.add_argument("--cells", default=",".join(CELLS))
    p.add_argument("--results-dir", default="sweeps/tqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ngpu", type=int, help="cap the number of GPUs used (default: all visible)")
    p.add_argument("--promote-cap", type=int, default=20,
                   help="max frontier points promoted per (cell, method)")
    p.add_argument("--only", help="comma list of tag substrings; run only matching jobs")
    p.add_argument("--folds", default="0,1",
                   help="comma list of CV folds to run (full jobs only; caps have no fold). "
                        "One job per invocation keeps peak RSS inside tight container "
                        "memory limits: activation tensors accumulated across jobs in one "
                        "process can trip the cgroup OOM killer.")
    p.add_argument("--iti-sigma", default="gen_end_q",
                   choices=["gen_end_q", "gen_end_q_qend"],
                   help="iti_sigma_position for the ITI grid (gen_end_q_qend = the "
                        "question-end sigma-read arm; cache keys differ, so the two "
                        "arms never collide)")
    p.add_argument("--list", action="store_true",
                   help="print the job plan with cache status, run nothing")
    args = p.parse_args()
    stages = set(args.stages.split(","))
    cells = [c for c in args.cells.split(",") if c in CELLS]

    folds = {int(f) for f in args.folds.split(",")}

    def keep(job):
        if args.only and not any(pat in job[1] for pat in args.only.split(",")):
            return False
        return job[0] != "full" or job[4] in folds

    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        round1 = [j for j in interleave(
            [list(full_jobs(cell, stages, args.iti_sigma)) for cell in cells]) if keep(j)]
        if args.list:
            for job in round1 + [j for j in interleave(
                    [list(cap_jobs(args, cell)) for cell in cells
                     if "caps" in stages]) if keep(j)]:
                state = "cached" if cached_metrics(job_overrides(args, job)) is not None \
                    else "pending"
                print(f"{job_desc(job)}\t{state}\t{' '.join(job_overrides(args, job))}")
            return

        execute(args, [j for j in round1 if cached_metrics(job_overrides(args, j)) is None])
        if stages & {"promote", "caps"}:
            harvest_fulls(args)  # barrier: promotion reads the harvested grid
        if "promote" in stages:
            promote(args)
            harvest_gate_density(args)
        if "caps" in stages:     # barrier: cap jobs come from promoted.tsv
            round2 = [j for j in interleave(
                [list(cap_jobs(args, cell)) for cell in cells]) if keep(j)]
            execute(args, [j for j in round2 if cached_metrics(job_overrides(args, j)) is None])
            harvest_caps(args)
    print(f"[{time.strftime('%H:%M')}] TQA SWEEP COMPLETE -> {args.results_dir}")


if __name__ == "__main__":
    sys.exit(main())

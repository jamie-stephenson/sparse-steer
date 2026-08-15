"""One-off heuristic-exploitation probe (question-blind control), not a pipeline stage.

A model that is genuinely more truthful needs the question; a model exploiting
TruthfulQA's surface regularities (generic/denial phrasing, length) does not. For every
swept config (all sparse + ITI grid points, plus the unsteered anchor) in every cell,
this evaluates the MISMATCHED pairing from the cached steering artifacts: each
question's MC answer set scored against the NEXT question in the eval split (fixed
rotation, identical for every config), and the truth judge rescoring the model's
generations against those rotated questions. Matched numbers are NOT rerun per config:
they already exist in the cache (fulls harvest), and the harvest joins them in.

Three SENTINEL jobs (one per method, spread across cells) DO rerun the matched eval
first, serially, and abort the whole run if any metric deviates from the cached grid
number by more than --tolerance: that validates that this probe's eval path reproduces
the engine that produced the paper numbers before any mismatched number is trusted.

A truth-tracking gain collapses toward chance under mismatch; a surface-preference
gain survives. Per-job JSON + generations TSVs land in <results-dir>/mismatch/
(resume = file existence); harvest writes mismatch.tsv.

  uv run python scripts/scratch/run_mismatch_probe.py --list
  uv run python scripts/scratch/run_mismatch_probe.py --folds 0
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tqa_experiments as rt

# (tag, fold): matched-eval reproduction checks, one per method, spanning base/chat
# models and plain/chat templates. Run first; any deviation beyond tolerance aborts.
SENTINELS = {("uns_base_qa", 0), ("sp_ll_ch_l0.005_open_ag", 0),
             ("iti_qw_qa_a15_k48_ag", 0)}
SENTINEL_KEYS = ["mc0", "mc1", "mc2", "gen_truthful", "gen_informative"]


def cell_jobs(args, cell):
    points = [(f"uns_{cell}", "unsteered", ["method=unsteered"])] + list(
        rt.grid_jobs(cell, args.iti_sigma))
    for fold in [int(f) for f in args.folds.split(",")]:
        for tag, method, cfg in points:
            yield (tag, cell, method, fold, cfg)


def jobs(args):
    # interleaved across cells so concurrent workers hold different models
    yield from rt.interleave([list(cell_jobs(args, c)) for c in args.cells.split(",")])


def out_paths(args, tag, fold):
    d = Path(args.results_dir) / "mismatch"
    return d / f"{tag}_f{fold}.json", d / f"{tag}_f{fold}_gens.tsv"


def run_probe(args, job, label="", strict=False):
    """Build the (cached) steered model for one grid config and run the mismatched
    eval (plus the matched eval on sentinel jobs). Mirrors Experiment.run()'s front
    half exactly (seeding, datasets, pipeline, eval-mode gates). With strict=True a
    failure (including a sentinel deviation) raises instead of being logged."""
    import csv

    import torch
    from sparse_steer.tasks.truthfulqa.eval import (
        TRUTH_JUDGE, _generate_answers, _judge_answers, evaluate)
    from sparse_steer.utils.compile import set_compile
    from sparse_steer.utils.memory import free_model_memory
    from sparse_steer.core.loading import load_tokenizer

    tag, cell, method, fold, cfg = job
    sentinel = (tag, fold) in SENTINELS
    res_path, gens_path = out_paths(args, tag, fold)
    print(f"[{time.strftime('%H:%M')}{label}] PROBE {tag} fold={fold}"
          + (" [sentinel]" if sentinel else ""), flush=True)
    exp = None
    model = None
    try:
        overrides = rt.full_overrides(args, cell, fold, cfg)
        exp = rt.compose_experiment(overrides)
        exp._seed_everything(exp.config.seed)
        set_compile(exp.config.get("compile_models", True))
        tokenizer = load_tokenizer(exp.config)
        extraction_ds, train_ds, eval_ds = exp.task.build_datasets(tokenizer, exp.config)
        model = exp._load_model()
        out_dir = Path("output") / "mismatch_probe" / f"{tag}_f{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model, _artifacts, _info = exp._run_pipeline(
            model, tokenizer, extraction_ds, train_ds, out_dir)
        model.eval()  # hard-concrete gates are stochastic in train mode

        spos = exp.config.get("steer_token_position", "all")
        template = exp.config.get("prompt_template", "chat")
        questions = [r["question"] for r in eval_ds]
        n = len(questions)
        # fixed rotation: question i scored/judged against question i+1's text. The eval
        # split order is deterministic per fold, so every config sees the same derangement.
        mm_questions = [questions[(i + 1) % n] for i in range(n)]
        ds_mm = eval_ds.map(
            lambda _ex, i: {"question": mm_questions[i]}, with_indices=True)

        mc_mm = evaluate(model, tokenizer, ds_mm, batch_size=exp.config.eval_batch_size,
                         steer_token_position=spos, template=template)
        answers = _generate_answers(
            model, tokenizer, questions,
            max_new_tokens=exp.config.get("gen_max_new_tokens", 64),
            batch_size=int(exp.config.get("gen_batch_size", 8)),
            steer_token_position=spos, template=template)
        jkw = dict(device=model.device,
                   batch_size=int(exp.config.get("judge_batch_size", 8)))
        truth_mm = _judge_answers(mm_questions, answers,
                                  judge_model_name=TRUTH_JUDGE, suffix="True", **jkw)
        res = {"mm_mc0": mc_mm["mc0"], "mm_mc1": mc_mm["mc1"], "mm_mc2": mc_mm["mc2"],
               "mm_gen_truthful": sum(truth_mm) / n, "n": n}

        truth = info = None
        if sentinel:
            from sparse_steer.tasks.truthfulqa.eval import INFO_JUDGE

            mc = evaluate(model, tokenizer, eval_ds,
                          batch_size=exp.config.eval_batch_size,
                          steer_token_position=spos, template=template)
            truth = _judge_answers(questions, answers,
                                   judge_model_name=TRUTH_JUDGE, suffix="True", **jkw)
            info = _judge_answers(questions, answers,
                                  judge_model_name=INFO_JUDGE, suffix="Helpful", **jkw)
            res.update(mc)
            res["gen_truthful"] = sum(truth) / n
            res["gen_informative"] = sum(info) / n
            cached = rt.cached_metrics(overrides)
            if cached is None:
                raise RuntimeError(f"sentinel {tag} f{fold}: no cached fulls metrics")
            drifts = {k: res[k] - cached[k] for k in SENTINEL_KEYS if k in cached}
            print(f"  sentinel drift vs cache: " + " ".join(
                f"{k}={d:+.4f}" for k, d in drifts.items()), flush=True)
            bad = {k: d for k, d in drifts.items() if abs(d) > args.tolerance}
            if bad:
                raise RuntimeError(
                    f"SENTINEL FAILED {tag} f{fold}: matched eval deviates from the "
                    f"cached grid numbers beyond {args.tolerance}: {bad}. The probe's "
                    "eval path does not reproduce the engine that produced the paper "
                    "numbers; aborting before any mismatched number is trusted.")

        res_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gens_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["question", "mm_question", "answer", "truthful_mm",
                        "truthful", "informative"])
            for i, (q, mq, a, tm) in enumerate(
                    zip(questions, mm_questions, answers, truth_mm)):
                w.writerow([q, mq, " ".join(a.split()), int(tm),
                            int(truth[i]) if truth else "", int(info[i]) if info else ""])
        res_path.write_text(json.dumps(res, indent=1))
        print(f"  {tag} f{fold}: mm_mc1 {res['mm_mc1']:.3f} "
              f"mm_true {res['mm_gen_truthful']:.3f}", flush=True)
    except Exception as e:  # no result file is written, so the job retries next run
        if strict:
            raise
        print(f"ERR {tag} f{fold}: {type(e).__name__}: {e}", flush=True)
    finally:
        del exp, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        free_model_memory()


def _worker(args, gpu, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # before any torch import
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        while (job := queue.get()) is not None:
            run_probe(args, job, label=f" gpu{gpu}")


def _sentinel_worker(args, gpu, sjobs):
    """Own spawned process so the GPU pin stays out of the parent (the parent's later
    gpu enumeration must still see every device)."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        for job in sjobs:
            run_probe(args, job, label=f" gpu{gpu}" if gpu else "", strict=True)


def execute(args, pending):
    gpus = rt.gpu_list(args)
    if len(gpus) <= 1:
        from hydra import initialize_config_dir

        with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
            for job in pending:
                run_probe(args, job)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for job in pending:
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
                print(f"WARNING: worker gpu{g} died (exit {w.exitcode}); respawning",
                      flush=True)
                workers[g] = ctx.Process(target=_worker, args=(args, g, queue))
                workers[g].start()


def harvest(args):
    """mismatch.tsv: one row per (tag, fold); matched columns joined from the cached
    fulls metrics, mismatched columns from the probe."""
    from hydra import initialize_config_dir

    out = Path(args.results_dir) / "mismatch.tsv"
    matched_keys = ["mc0", "mc1", "mc2", "gen_truthful", "gen_informative"]
    mm_keys = ["mm_mc0", "mm_mc1", "mm_mc2", "mm_gen_truthful"]
    rows = []
    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        for tag, cell, method, fold, cfg in jobs(args):
            res_path, _ = out_paths(args, tag, fold)
            if not res_path.exists():
                continue
            res = json.loads(res_path.read_text())
            cached = rt.cached_metrics(rt.full_overrides(args, cell, fold, cfg)) or {}
            rows.append("\t".join(
                [tag, cell, method, str(fold)]
                + [f"{cached[k]:.4f}" if k in cached else "--" for k in matched_keys]
                + [f"{res[k]:.4f}" for k in mm_keys]))
    hdr = ("tag\tcell\tmethod\tfold\t" + "\t".join(matched_keys) + "\t"
           + "\t".join(mm_keys) + "\n")
    out.write_text(hdr + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} probe rows -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="sweeps/tqa")
    p.add_argument("--cells", default=",".join(rt.CELLS))
    p.add_argument("--folds", default="0,1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ngpu", type=int)
    p.add_argument("--iti-sigma", default="gen_end_q")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="max |matched - cached| on any sentinel metric before aborting")
    p.add_argument("--only", help="comma list of tag substrings; run only matching jobs")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    plan = [j for j in jobs(args)
            if not args.only or any(pat in j[0] for pat in args.only.split(","))]
    pending = [j for j in plan if not out_paths(args, j[0], j[3])[0].exists()]
    if args.list:
        for tag, _cell, _method, fold, _cfg in plan:
            done = out_paths(args, tag, fold)[0].exists()
            mark = " [sentinel]" if (tag, fold) in SENTINELS else ""
            print(f"PROBE {tag} fold={fold}{mark}\t{'done' if done else 'pending'}")
        print(f"{len(plan) - len(pending)} done, {len(pending)} pending")
        return

    # sentinels first, serially and strictly: a deviation aborts before the fleet runs
    sentinel_jobs = [j for j in pending if (j[0], j[3]) in SENTINELS]
    if sentinel_jobs:
        import multiprocessing as mp

        gpus = rt.gpu_list(args)
        ctx = mp.get_context("spawn")
        w = ctx.Process(target=_sentinel_worker,
                        args=(args, gpus[0] if gpus else None, sentinel_jobs))
        w.start()
        w.join()
        if w.exitcode != 0:
            sys.exit(f"sentinel phase failed (exit {w.exitcode}); "
                     "not running any mismatched jobs")
        print("all sentinels reproduce the cached grid numbers", flush=True)

    execute(args, [j for j in pending if (j[0], j[3]) not in SENTINELS])
    harvest(args)
    print(f"[{time.strftime('%H:%M')}] MISMATCH PROBE COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

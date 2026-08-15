"""One-off heuristic-exploitation probe (question-blind control), not a pipeline stage.

A model that is genuinely more truthful needs the question; a model exploiting
TruthfulQA's surface regularities (generic/denial phrasing, length) does not. For every
swept config (all sparse + ITI grid points, plus the unsteered anchor) in every cell,
this reruns the full TQA eval twice per fold from the cached steering artifacts:

  matched     the standard pairing (sanity: must reproduce the cached fulls numbers)
  mismatched  each question's MC answer set scored against the NEXT question in the
              eval split (fixed rotation, identical for every config), and the truth
              judge rescoring the SAME generations against those rotated questions

A truth-tracking gain collapses toward chance under mismatch; a surface-preference
gain survives. Steering models come straight from the cache via the exact grid
configs (no retraining). Per-job JSON + generations TSVs land in
<results-dir>/mismatch/ (resume = file existence); harvest writes mismatch.tsv.

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


def run_probe(args, job, label=""):
    """Build the (cached) steered model for one grid config and run the matched +
    mismatched evals. Mirrors Experiment.run()'s front half exactly (seeding, datasets,
    pipeline, eval-mode gates), then swaps the standard eval for the probe."""
    import csv

    import torch
    from sparse_steer.tasks.truthfulqa.eval import (
        INFO_JUDGE, TRUTH_JUDGE, _generate_answers, _judge_answers, evaluate)
    from sparse_steer.utils.compile import set_compile
    from sparse_steer.utils.memory import free_model_memory
    from sparse_steer.utils.models import load_tokenizer

    tag, cell, method, fold, cfg = job
    res_path, gens_path = out_paths(args, tag, fold)
    print(f"[{time.strftime('%H:%M')}{label}] PROBE {tag} fold={fold}", flush=True)
    exp = None
    model = None
    try:
        exp = rt.compose_experiment(rt.full_overrides(args, cell, fold, cfg))
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

        mc = evaluate(model, tokenizer, eval_ds, batch_size=exp.config.eval_batch_size,
                      steer_token_position=spos, template=template)
        mc_mm = evaluate(model, tokenizer, ds_mm, batch_size=exp.config.eval_batch_size,
                         steer_token_position=spos, template=template)

        answers = _generate_answers(
            model, tokenizer, questions,
            max_new_tokens=exp.config.get("gen_max_new_tokens", 64),
            batch_size=int(exp.config.get("gen_batch_size", 8)),
            steer_token_position=spos, template=template)
        jkw = dict(device=model.device,
                   batch_size=int(exp.config.get("judge_batch_size", 8)))
        truth = _judge_answers(questions, answers,
                               judge_model_name=TRUTH_JUDGE, suffix="True", **jkw)
        info = _judge_answers(questions, answers,
                              judge_model_name=INFO_JUDGE, suffix="Helpful", **jkw)
        truth_mm = _judge_answers(mm_questions, answers,
                                  judge_model_name=TRUTH_JUDGE, suffix="True", **jkw)

        res_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gens_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["question", "mm_question", "answer", "truthful", "informative",
                        "truthful_mm"])
            for q, mq, a, t, i, tm in zip(questions, mm_questions, answers, truth, info,
                                          truth_mm):
                w.writerow([q, mq, " ".join(a.split()), int(t), int(i), int(tm)])
        res = {**mc,
               "mm_mc0": mc_mm["mc0"], "mm_mc1": mc_mm["mc1"], "mm_mc2": mc_mm["mc2"],
               "gen_truthful": sum(truth) / n, "gen_informative": sum(info) / n,
               "gen_truthful_informative": sum(t and i for t, i in zip(truth, info)) / n,
               "mm_gen_truthful": sum(truth_mm) / n, "n": n}
        res_path.write_text(json.dumps(res, indent=1))
        print(f"  {tag} f{fold}: mc1 {res['mc1']:.3f} -> mm {res['mm_mc1']:.3f}, "
              f"true {res['gen_truthful']:.3f} -> mm {res['mm_gen_truthful']:.3f}", flush=True)
    except Exception as e:  # no result file is written, so the job retries next run
        print(f"ERR {tag} f{fold}: {type(e).__name__}: {e}", flush=True)
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
    """mismatch.tsv: one row per (tag, fold), matched + mismatched metrics, plus the
    matched-vs-cached-fulls mc1 delta as the reproduction sanity column."""
    from hydra import initialize_config_dir

    out = Path(args.results_dir) / "mismatch.tsv"
    cols = ["mc0", "mc1", "mc2", "mm_mc0", "mm_mc1", "mm_mc2", "gen_truthful",
            "gen_informative", "gen_truthful_informative", "mm_gen_truthful"]
    rows = []
    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        for tag, cell, method, fold, cfg in jobs(args):
            res_path, _ = out_paths(args, tag, fold)
            if not res_path.exists():
                continue
            res = json.loads(res_path.read_text())
            cached = rt.cached_metrics(rt.full_overrides(args, cell, fold, cfg)) or {}
            drift = (f"{res['mc1'] - cached['mc1']:+.4f}" if "mc1" in cached else "--")
            rows.append("\t".join(
                [tag, cell, method, str(fold)] + [f"{res[c]:.4f}" for c in cols] + [drift]))
    hdr = "tag\tcell\tmethod\tfold\t" + "\t".join(cols) + "\tmc1_vs_cached\n"
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
    p.add_argument("--only", help="comma list of tag substrings; run only matching jobs")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    plan = [j for j in jobs(args)
            if not args.only or any(pat in j[0] for pat in args.only.split(","))]
    pending = [j for j in plan if not out_paths(args, j[0], j[3])[0].exists()]
    if args.list:
        for tag, cell, _method, fold, _cfg in plan:
            done = out_paths(args, tag, fold)[0].exists()
            print(f"PROBE {tag} fold={fold}\t{'done' if done else 'pending'}")
        print(f"{len(plan) - len(pending)} done, {len(pending)} pending")
        return
    execute(args, pending)
    harvest(args)
    print(f"[{time.strftime('%H:%M')}] MISMATCH PROBE COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

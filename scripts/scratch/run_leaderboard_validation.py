"""One-off sanity check of the evaluation stack (appendix table tab:val-caps): the
unsteered models scored once under the PUBLIC few-shot protocols on the full benchmarks,
five-shot MMLU and twenty-five-shot ARC, where published reference values exist (Open
LLM Leaderboard for the two Llama models, the release report for Qwen's MMLU). The
committed capability suite stays zero-shot on subsamples; this run only anchors the
stack against public numbers.

  uv run python scripts/scratch/run_leaderboard_validation.py --list
  uv run python scripts/scratch/run_leaderboard_validation.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tqa_experiments as rt

# (cell, bench tag, lmeval overrides); Qwen ARC has no published figure, so it is not run
JOBS = [
    (cell, bench, ["method=unsteered", f"lmeval_tasks=[{task}]",
                   f"lmeval_fewshot={shots}", "lmeval_steer=all"])
    for cell, benches in (("base_qa", ("mmlu", "arc")), ("ll_qa", ("mmlu", "arc")),
                          ("qw_qa", ("mmlu",)))
    for bench, task, shots in ((("mmlu", "mmlu", 5),) if benches == ("mmlu",) else
                               (("mmlu", "mmlu", 5), ("arc", "arc_challenge", 25)))
    if bench in benches
]


def jobs():
    for cell, bench, cfg in JOBS:
        yield ("cap", f"cap_lb{bench}_{cell}_uns", cell, "unsteered",
               f"leaderboard-{bench}", cfg)


def harvest(args):
    out = Path(args.results_dir) / "leaderboard_validation.tsv"
    rows = []
    for _, tag, cell, method, stage, cfg in jobs():
        m = rt.cached_metrics(rt.cap_overrides(args, cell, cfg))
        if m is None:
            continue
        metrics = " ".join(f"{k.upper()}: {v:.4f}" for k, v in sorted(m.items())
                           if isinstance(v, (int, float))
                           and k.upper().startswith(("MMLU", "ARC")))
        rows.append(f"{tag}\t{cell}\t{method}\t{stage}\t{metrics}")
    out.write_text(rt.CAPS_HDR + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} rows -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="sweeps/tqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ngpu", type=int)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        plan = list(jobs())
        if args.list:
            for job in plan:
                state = ("cached" if rt.cached_metrics(rt.job_overrides(args, job))
                         is not None else "pending")
                print(f"{rt.job_desc(job)}\t{state}")
            return
        rt.execute(args, [j for j in plan
                          if rt.cached_metrics(rt.job_overrides(args, j)) is None])
        harvest(args)
    print(f"[{time.strftime('%H:%M')}] LEADERBOARD VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

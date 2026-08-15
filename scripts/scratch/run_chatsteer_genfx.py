"""One-off probe, not part of the committed caps protocol: chat-template-extracted
steering evaluated under the FIXED-template generative capability evals.

The committed caps stage (run_tqa_experiments.cap_jobs) gives chat cells the
chat-template generative evals and plain cells the fixed-template ones, so the grid
never measures chat-extracted steering against the fixed-format MMLU/ARC contract.
This driver fills that cell: for ll_ch and qw_ch (Llama base has no chat extraction),
every grid config (all 20 sparse + 18 ITI, not just the frontier) plus the unsteered
anchor gets generative MMLU + ARC with inspect_apply_template=false, steering position
following the config's own steer_token_position, limits identical to the committed
caps stage. Steering/extraction artifacts are read from the existing cache, so jobs
are eval-only; results land in the cache and harvest to <results-dir>/genfx_chatsteer.tsv.

  uv run python scripts/scratch/run_chatsteer_genfx.py --list     # plan + cache status
  uv run python scripts/scratch/run_chatsteer_genfx.py            # run all GPUs + harvest
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tqa_experiments as rt

CELLS = ["ll_ch", "qw_ch"]


def jobs(args, cell):
    points = [("uns", "unsteered", ["method=unsteered"])] + list(
        rt.grid_jobs(cell, args.iti_sigma))
    for ptag, method, cfg in points:
        pos = rt._cfg_steer_pos(cfg)
        gen_extra = [f"inspect_steer={pos}", "inspect_apply_template=false"]
        yield ("cap", f"cap_mmgenfx_{cell}_{ptag}", cell, method, "generative-fx-mmlu",
               cfg + ["inspect_evals=[mmlu]"] + rt._GEN + gen_extra)
        yield ("cap", f"cap_arcgenfx_{cell}_{ptag}", cell, method, "generative-fx-arc",
               cfg + ["inspect_evals=[arc_challenge]"] + rt._GEN + gen_extra)


def harvest(args):
    out = Path(args.results_dir) / "genfx_chatsteer.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in CELLS:
        for _, tag, cell_, method, stage, cfg in jobs(args, cell):
            m = rt.cached_metrics(rt.cap_overrides(args, cell_, cfg))
            if m is None:
                continue
            metrics = " ".join(f"{k.upper()}: {v:.4f}" for k, v in sorted(m.items())
                               if isinstance(v, (int, float))
                               and k.upper().startswith(("MMLU", "ARC")))
            rows.append(f"{tag}\t{cell_}\t{method}\t{stage}\t{metrics}")
    out.write_text(rt.CAPS_HDR + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} cached rows -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="sweeps/tqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ngpu", type=int)
    p.add_argument("--iti-sigma", default="gen_end_q")
    p.add_argument("--only", help="comma list of tag substrings; run only matching jobs")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    def keep(job):
        return not args.only or any(pat in job[1] for pat in args.only.split(","))

    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        plan = [j for j in rt.interleave([list(jobs(args, c)) for c in CELLS]) if keep(j)]
        if args.list:
            for job in plan:
                state = ("cached" if rt.cached_metrics(rt.job_overrides(args, job))
                         is not None else "pending")
                print(f"{rt.job_desc(job)}\t{state}")
            return
        rt.execute(args, [j for j in plan
                          if rt.cached_metrics(rt.job_overrides(args, j)) is None])
        harvest(args)
    print(f"[{time.strftime('%H:%M')}] CHATSTEER GENFX COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

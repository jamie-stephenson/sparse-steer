"""Run the TruthfulQA grid: train + evaluate config-folds across the visible GPUs (one in-process
grid_runner per GPU, so the judges load once per GPU), then aggregate to fulls.tsv + grid_2fold.tsv.

This is the GRID stage only -- no capability suite, no Pareto promote (caps is scripts/run_caps.sh).
WHICH configs to run is chosen by CLI arguments, not environment variables:

  uv run python scripts/run_grid.py                        # full grid: l0 in {0,0.005,0.01,0.03} + ITI + unsteered
  uv run python scripts/run_grid.py --only-sparse-l0 0.03  # only the l0=0.03 sparse configs (extend a cached grid)
  uv run python scripts/run_grid.py --cells qw_ch,ll_ch    # a subset of cells
  uv run python scripts/run_grid.py --ngpu 2 --results-dir sweeps/foo

Assumes HF_TOKEN / HF_HOME / PYTORCH_CUDA_ALLOC_CONF etc. are already exported by the caller
(scripts/reproduce.sh does this). Resumable: grid_runner skips any (tag,cell,method,fold) already
present in a shard's fulls.tsv, so a re-run only trains what is missing.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import emit_grid_jobs as G  # noqa: E402


def detect_ngpu():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", default="ll_qa,ll_ch,qw_qa,qw_ch,base_qa",
                   help="comma-separated cells (model x template) to run")
    p.add_argument("--only-sparse-l0", default=None,
                   help="restrict to sparse configs at this single l0_lambda (skip unsteered/ITI/other l0)")
    p.add_argument("--results-dir", default="sweeps/v2", help="output dir for shards + fulls/grid_2fold")
    p.add_argument("--ngpu", type=int, default=None, help="GPUs to use (default: all visible)")
    a = p.parse_args()

    ngpu = a.ngpu or detect_ngpu()
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    cells = a.cells.split(",")

    rows = G.build_jobs(cells, only_sparse_l0=a.only_sparse_l0)
    label = f"l0_{a.only_sparse_l0}" if a.only_sparse_l0 else "all"
    jobs_path = res / f"{label}.jobs"
    jobs_path.write_text("".join("\t".join(r) + "\n" for r in rows))
    scope = f"only sparse l0={a.only_sparse_l0}" if a.only_sparse_l0 else "full grid (l0 {0,0.005,0.01,0.03} + ITI + unsteered)"
    print(f"[run_grid] {scope}: {len(rows)} config-folds across {ngpu} gpu(s) -> {jobs_path}", flush=True)

    # round-robin config-folds across GPUs; each grid_runner appends to its own shard's fulls.tsv
    procs = []
    for g in range(ngpu):
        shard = res / f"g{g}.jobs"
        shard.write_text("".join("\t".join(r) + "\n" for i, r in enumerate(rows) if i % ngpu == g))
        d = res / f"grid_g{g}"
        d.mkdir(exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        log = open(f"/tmp/grid_g{g}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "grid_runner.py"), str(d), str(shard)],
            env=env, stdout=log, stderr=subprocess.STDOUT))
    failed = [g for g, pr in enumerate(procs) if pr.wait() != 0]
    if failed:
        print(f"[run_grid] WARNING grid_runner nonzero exit on gpu(s) {failed} (see /tmp/grid_g*.log)", flush=True)

    # merge shards -> fulls.tsv, then fold-mean (one row per config, fold-0 args). No promote.
    fulls = res / "fulls.tsv"
    with open(fulls, "w") as out:
        wrote_hdr = False
        for g in range(ngpu):
            f = res / f"grid_g{g}" / "fulls.tsv"
            if not f.exists():
                continue
            lines = f.read_text().splitlines()
            if not wrote_hdr:
                out.write(lines[0] + "\n")
                wrote_hdr = True
            for ln in lines[1:]:
                out.write(ln + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sweep_fold_mean.py"),
                    str(fulls), str(res / "grid_2fold.tsv")], check=True)
    ncfg = len((res / "grid_2fold.tsv").read_text().splitlines()) - 1
    print(f"[run_grid] === GRID COMPLETE === fulls:{fulls} configs:{ncfg}", flush=True)


if __name__ == "__main__":
    main()

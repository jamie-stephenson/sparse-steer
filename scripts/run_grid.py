"""Run the TruthfulQA sweep: train + evaluate the grid across all visible GPUs, then run the capability
suite on every trained config. With no flags this reproduces the whole study (multi-day). Each grid
dimension can be narrowed independently; unspecified dimensions keep their full default set:

  uv run python scripts/run_grid.py                              # full grid + caps (the reproduction)
  uv run python scripts/run_grid.py --l0 0.03 --method sparse    # only l0=0.03 sparse cells, then cap them
  uv run python scripts/run_grid.py --model qwen llama-base      # only those models (all other dims default)
  uv run python scripts/run_grid.py --method iti --pos ag        # only ITI, answer_gen position

Stages (both resumable -- re-running only does missing work):
  GRID  one in-process grid_runner per GPU (judges load once/GPU) -> fulls.tsv, grid_2fold.tsv
  CAPS  one caps_runner per GPU over EVERY config of the touched cells (no promote), reusing every caps
        row already computed anywhere under the results dir. Skip with --skip-caps.

Assumes HF_TOKEN / HF_HOME / PYTORCH_CUDA_ALLOC_CONF etc. are exported by the caller.
"""
import argparse
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


def _launch_per_gpu(cmd_for_gpu, ngpu, logfile_for_gpu):
    """Run one subprocess per GPU (CUDA_VISIBLE_DEVICES set), wait, return list of GPUs that failed."""
    import os
    procs = []
    for g in range(ngpu):
        cmd = cmd_for_gpu(g)
        if cmd is None:
            procs.append(None)
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(logfile_for_gpu(g), "w"),
                                      stderr=subprocess.STDOUT))
    return [g for g, pr in enumerate(procs) if pr is not None and pr.wait() != 0]


def run_grid(rows, res, ngpu):
    label = "grid"
    jobs = res / f"{label}.jobs"
    jobs.write_text("".join("\t".join(r) + "\n" for r in rows))
    print(f"[run_grid] GRID: {len(rows)} config-folds across {ngpu} gpu(s)", flush=True)
    shard_files = []
    for g in range(ngpu):
        sf = res / f"g{g}.jobs"
        sf.write_text("".join("\t".join(r) + "\n" for i, r in enumerate(rows) if i % ngpu == g))
        shard_files.append(sf)
        (res / f"grid_g{g}").mkdir(exist_ok=True)

    def cmd(g):
        if shard_files[g].read_text().strip() == "":
            return None
        return [sys.executable, str(ROOT / "scripts" / "grid_runner.py"),
                str(res / f"grid_g{g}"), str(shard_files[g])]
    failed = _launch_per_gpu(cmd, ngpu, lambda g: f"/tmp/grid_g{g}.log")
    if failed:
        print(f"[run_grid] WARNING grid_runner nonzero exit on gpu(s) {failed}", flush=True)

    fulls = res / "fulls.tsv"
    with open(fulls, "w") as out:
        wrote = False
        for g in range(ngpu):
            f = res / f"grid_g{g}" / "fulls.tsv"
            if not f.exists():
                continue
            lines = f.read_text().splitlines()
            if not wrote:
                out.write(lines[0] + "\n")
                wrote = True
            for ln in lines[1:]:
                out.write(ln + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sweep_fold_mean.py"),
                    str(fulls), str(res / "grid_2fold.tsv")], check=True)
    print(f"[run_grid] GRID done -> {res}/grid_2fold.tsv", flush=True)


CAPS_HDR = "tag\tcell\tmethod\tstage\tmetrics\n"


def run_caps(rows, res, ngpu):
    """Cap ONLY the configs this run's grid covered (rows), not every config of the touched cells.
    Prior caps rows are reused; the cell->shard mapping is stable so a subset run never clobbers other
    cells' cached caps."""
    g2f = res / "grid_2fold.tsv"
    if not g2f.exists():
        print("[run_grid] no grid_2fold.tsv; nothing to cap", flush=True)
        return
    lines = g2f.read_text().splitlines()
    hdr, cols = lines[0], lines[0].split("\t")
    ti, ci = cols.index("tag"), cols.index("cell")
    trained = {(f[ci], f[ti]): ln for ln in lines[1:] for f in [ln.split("\t")]}
    want = sorted({(r[1], r[0]) for r in rows if (r[1], r[0]) in trained})  # (cell, tag), trained only
    if not want:
        print("[run_grid] CAPS: none of this run's configs are trained yet", flush=True)
        return

    # pool every caps row already computed anywhere (master + targeted dirs), dedup by tag
    known = {}
    for f in [res / "caps.tsv", *sorted(res.glob("cap*/caps.tsv"))]:
        if f.exists():
            for ln in f.read_text().splitlines()[1:]:
                known.setdefault(ln.split("\t", 1)[0], ln)

    all_cells = G.cells_for()   # canonical order -> stable cell->shard
    shard_of = {c: all_cells.index(c) % ngpu for c in all_cells}
    shard_cells = {g: sorted({c for (c, t) in want if shard_of[c] == g}) for g in range(ngpu)}
    print(f"[run_grid] CAPS: {len(want)} configs over {len({c for c, _ in want})} cell(s)", flush=True)

    def cmd(g):
        cs = shard_cells[g]
        if not cs:
            return None
        d = res / f"cap_g{g}"
        d.mkdir(exist_ok=True)
        with open(d / "caps.tsv", "w") as f:            # seed all known caps -> already-done skip
            f.write(CAPS_HDR)
            f.writelines(ln + "\n" for ln in known.values())
        with open(d / "promoted.tsv", "w") as f:        # config list = ONLY this run's configs here
            f.write(hdr + "\n")
            f.writelines(trained[(c, t)] + "\n" for (c, t) in want if c in cs)
        return [sys.executable, str(ROOT / "scripts" / "caps_runner.py"), str(d), ",".join(cs)]
    failed = _launch_per_gpu(cmd, ngpu, lambda g: f"/tmp/v2_caps_g{g}.log")
    if failed:
        print(f"[run_grid] WARNING caps_runner nonzero exit on gpu(s) {failed}", flush=True)

    merged = {}
    for f in sorted(res.glob("cap_g*/caps.tsv")):
        for ln in f.read_text().splitlines()[1:]:
            merged.setdefault(ln.split("\t", 1)[0], ln)
    with open(res / "caps.tsv", "w") as out:
        out.write(CAPS_HDR)
        out.writelines(ln + "\n" for ln in merged.values())
    print(f"[run_grid] CAPS done -> {res}/caps.tsv ({len(merged)} rows)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", choices=G.DIMS["model"], metavar="M")
    p.add_argument("--template", nargs="+", choices=G.DIMS["template"], metavar="T")
    p.add_argument("--method", nargs="+", choices=G.DIMS["method"], metavar="M")
    p.add_argument("--l0", nargs="+", choices=G.DIMS["l0"], metavar="L")
    p.add_argument("--init", nargs="+", choices=G.DIMS["init"], metavar="I")
    p.add_argument("--pos", nargs="+", choices=G.DIMS["pos"], metavar="P")
    p.add_argument("--iti-scale", nargs="+", choices=G.DIMS["iti_scale"], metavar="A")
    p.add_argument("--iti-k", nargs="+", choices=G.DIMS["iti_k"], metavar="K")
    p.add_argument("--fold", nargs="+", choices=G.DIMS["fold"], metavar="F")
    p.add_argument("--results-dir", default="sweeps/v2")
    p.add_argument("--ngpu", type=int, default=None)
    p.add_argument("--skip-caps", action="store_true", help="run the grid only, no capability suite")
    a = p.parse_args()

    ngpu = a.ngpu or detect_ngpu()
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    rows = G.build_jobs(models=a.model, templates=a.template, methods=a.method, l0s=a.l0,
                        inits=a.init, positions=a.pos, iti_scales=a.iti_scale, iti_ks=a.iti_k, folds=a.fold)
    run_grid(rows, res, ngpu)
    if not a.skip_caps:
        run_caps(rows, res, ngpu)
    print("[run_grid] === COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

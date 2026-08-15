"""Post-hoc info-judge pass over the mismatch probe's saved generations.

The truth judge passes irrelevant answers as truthful, so the question-sensitivity of
the generation-side control lives in the informativeness judge. For every generations
file the probe saved (scripts/scratch/run_mismatch_probe.py), score every answer with
the info judge twice, against its own question and against the rotated one, and store
the per-row bits plus rates. Pure judging over saved text: no steering, no task models.

  uv run python scripts/scratch/run_info_mm.py            # all GPUs, resume by file
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def score_file(gens_path: Path, batch: int):
    import torch
    from sparse_steer.tasks.truthfulqa.eval import INFO_JUDGE, _judge_answers

    out = gens_path.with_name(gens_path.name.replace("_gens.tsv", "_info.json"))
    rows = list(csv.DictReader(open(gens_path), delimiter="\t"))
    if not rows:
        print(f"SKIP empty {gens_path.name}", flush=True)
        return
    q = [r["question"] for r in rows]
    mq = [r["mm_question"] for r in rows]
    a = [r["answer"] for r in rows]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    info = _judge_answers(q, a, judge_model_name=INFO_JUDGE, suffix="Helpful",
                          device=dev, batch_size=batch)
    info_mm = _judge_answers(mq, a, judge_model_name=INFO_JUDGE, suffix="Helpful",
                             device=dev, batch_size=batch)
    n = len(rows)
    out.write_text(json.dumps({
        "info": sum(info) / n, "info_mm": sum(info_mm) / n, "n": n,
        "bits_info": "".join(str(int(b)) for b in info),
        "bits_info_mm": "".join(str(int(b)) for b in info_mm)}))
    print(f"[{time.strftime('%H:%M')}] {gens_path.name}: info {sum(info)/n:.3f} "
          f"info_mm {sum(info_mm)/n:.3f}", flush=True)


def _worker(gpu, batch, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # before any torch import
    while (path := queue.get()) is not None:
        try:
            score_file(Path(path), batch)
        except Exception as e:
            print(f"ERR {path}: {type(e).__name__}: {e}", flush=True)


def harvest(mdir: Path):
    out = mdir.parent / "mismatch_info.tsv"
    rows = []
    for f in sorted(mdir.glob("*_info.json")):
        tag_fold = f.name[: -len("_info.json")]
        tag, fold = tag_fold.rsplit("_f", 1)
        d = json.loads(f.read_text())
        rows.append(f"{tag}\t{fold}\t{d['info']:.4f}\t{d['info_mm']:.4f}\t{d['n']}")
    out.write_text("tag\tfold\tinfo\tinfo_mm\tn\n" + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} info rows -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="sweeps/tqa")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ngpu", type=int)
    args = p.parse_args()
    mdir = Path(args.results_dir) / "mismatch"
    pending = [f for f in sorted(mdir.glob("*_gens.tsv"))
               if not f.with_name(f.name.replace("_gens.tsv", "_info.json")).exists()]
    print(f"{len(pending)} files pending", flush=True)
    gpus = []
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        gpus = [x for x in env.split(",") if x]
    else:
        try:
            import subprocess
            n = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                   text=True).stdout.strip().splitlines())
            gpus = [str(i) for i in range(n)]
        except Exception:
            gpus = []
    if args.ngpu:
        gpus = gpus[: args.ngpu]
    if len(gpus) <= 1:
        for f in pending:
            score_file(f, args.batch)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for f in pending:
            queue.put(str(f))
        for _ in gpus:
            queue.put(None)
        workers = [ctx.Process(target=_worker, args=(g, args.batch, queue)) for g in gpus]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    harvest(mdir)
    print(f"[{time.strftime('%H:%M')}] INFO MM PASS COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

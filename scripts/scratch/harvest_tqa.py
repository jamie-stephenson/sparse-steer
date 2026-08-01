"""Collate tqa sweep shard outputs into master plot-ready tables.

Reads one or more RESULTS_DIR paths produced by scripts/sweep_tqa.sh (each holding
screens.tsv / fulls.tsv / caps.tsv) and writes, to --out:

  master_frontier.tsv  one row per (cell, method, tag): 2-fold True/Info/MC1/MC2
                       (mean of fold 0/1) plus the per-fold values. Unsteered rows
                       included, so every frontier plot reads from this one file.
  master_caps.tsv      one row per (cell, method, tag, stage, metric): value plus
                       delta vs the same cell/stage/metric unsteered row.

Usage: uv run python scripts/harvest_tqa.py sweeps/tqa_g0 sweeps/tqa_g1 sweeps/tqa_base --out results/tqa
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def read_tsv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/tqa")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── frontier: fold pairs -> 2-fold means ────────────────────────────────
    folds = defaultdict(dict)  # (cell, method, tag) -> {fold: row}
    for d in args.dirs:
        for r in read_tsv(Path(d) / "fulls.tsv"):
            folds[(r["cell"], r["method"], r["tag"])][r["fold"]] = r

    with open(out / "master_frontier.tsv", "w") as f:
        cols = ["cell", "method", "tag", "true", "info", "mc1", "mc2",
                "true_f0", "info_f0", "mc1_f0", "mc2_f0",
                "true_f1", "info_f1", "mc1_f1", "mc2_f1"]
        f.write("\t".join(cols) + "\n")
        for (cell, method, tag), by_fold in sorted(folds.items()):
            def val(fold, k):
                r = by_fold.get(fold)
                return float(r[k]) if r and r.get(k) else None
            row = [cell, method, tag]
            for k in ("true", "info", "mc1", "mc2"):
                f0, f1 = val("0", k), val("1", k)
                two = (f0 + f1) / 2 if f0 is not None and f1 is not None else None
                row.append(f"{two:.4f}" if two is not None else "")
            for fold in ("0", "1"):
                for k in ("true", "info", "mc1", "mc2"):
                    v = val(fold, k)
                    row.append(f"{v:.4f}" if v is not None else "")
            f.write("\t".join(row) + "\n")

    # ── caps: explode metric strings, attach deltas vs unsteered ────────────
    metric_re = re.compile(r"([A-Z_/]+): ([0-9.]+)")
    rows = []
    for d in args.dirs:
        for r in read_tsv(Path(d) / "caps.tsv"):
            for metric, v in metric_re.findall(r.get("metrics", "")):
                if metric.endswith(("SAMPLE_LEN", "STDERR")):
                    continue
                rows.append({"cell": r["cell"], "method": r["method"], "tag": r["tag"],
                             "stage": r["stage"], "metric": metric, "value": float(v)})
    uns = {(r["cell"], r["stage"], r["metric"]): r["value"]
           for r in rows if r["method"] == "unsteered"}
    with open(out / "master_caps.tsv", "w") as f:
        f.write("cell\tmethod\ttag\tstage\tmetric\tvalue\tdelta_vs_unsteered\n")
        for r in sorted(rows, key=lambda x: (x["cell"], x["stage"], x["metric"], x["method"], x["tag"])):
            base = uns.get((r["cell"], r["stage"], r["metric"]))
            delta = f"{r['value'] - base:+.4f}" if base is not None and r["method"] != "unsteered" else ""
            f.write(f"{r['cell']}\t{r['method']}\t{r['tag']}\t{r['stage']}\t{r['metric']}\t{r['value']:.4f}\t{delta}\n")

    n_front = sum(1 for _ in open(out / "master_frontier.tsv")) - 1
    n_caps = sum(1 for _ in open(out / "master_caps.tsv")) - 1
    print(f"wrote {out}/master_frontier.tsv ({n_front} rows), {out}/master_caps.tsv ({n_caps} rows)")


if __name__ == "__main__":
    main()

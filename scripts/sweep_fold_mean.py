"""Collapse a 2-fold fulls.tsv into one 2-fold-mean row per config, in the screen-TSV column
layout (tag / cell / method / true / info / mc1 / mc2 / args) that scripts/sweep_promote.py expects.

Only configs with BOTH folds' True and Info present are emitted, so the Pareto promotion always
runs on complete 2-fold means (a half-finished config is simply not yet promotable — resumable).

Usage: uv run python scripts/sweep_fold_mean.py sweeps/tqa/fulls.tsv sweeps/tqa/grid_2fold.tsv
"""
import csv
import sys
from collections import defaultdict

inp, out = sys.argv[1], sys.argv[2]
groups = defaultdict(list)
for r in csv.DictReader(open(inp), delimiter="\t"):
    if r["method"] == "unsteered":
        continue
    # Group by config identity ONLY — never include args, which carries fold=N, so keying on it
    # would split a config's two folds into separate groups and drop every config (len(have)<2).
    groups[(r["tag"], r["cell"], r["method"])].append(r)

with open(out, "w") as f:
    f.write("tag\tcell\tmethod\ttrue\tinfo\tmc1\tmc2\targs\n")
    for (tag, cell, method), rs in sorted(groups.items()):
        folds = {r["fold"]: r for r in rs}
        def mean(k):
            vs = [float(folds[fd][k]) for fd in folds if folds[fd].get(k)]
            return sum(vs) / len(vs) if vs else None
        # require both folds to have contributed True and Info
        have = [fd for fd in folds if folds[fd].get("true") and folds[fd].get("info")]
        if len(have) < 2:
            continue
        # caps runs capability once per config on one fold's trained gates; use fold 0's args verbatim.
        args = folds.get("0", rs[0])["args"]
        t, i, m1, m2 = mean("true"), mean("info"), mean("mc1"), mean("mc2")
        f.write(f"{tag}\t{cell}\t{method}\t{t:.4f}\t{i:.4f}\t"
                f"{m1 if m1 is not None else ''}\t{m2 if m2 is not None else ''}\t{args}\n")

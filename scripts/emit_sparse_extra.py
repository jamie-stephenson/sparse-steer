"""Set up a targeted capability run over the GENUINELY-SPARSE non-promoted configs (POST-SWEEP item 6
follow-up). The promoted frontier (Pareto on True*Info) over-selected dense l0_lambda=0 configs, so
caps.tsv lacks capability numbers for the low-density end. This emits, per GPU shard, a promoted.tsv
of the sparse configs with l0_lambda>=0.005 that were NOT promoted, so caps_runner.py can measure them
and complete the density-vs-capability curve. Unsteered rows are pre-seeded from the main caps.tsv so
they are skipped (not recomputed).

Usage:  uv run python scripts/emit_sparse_extra.py <results_dir>   (reads grid_2fold.tsv, promoted.tsv,
        caps.tsv; writes <dir>/cap_sparse_g{0,1,2}/{promoted.tsv,caps.tsv})
"""
import csv
import re
import sys
from pathlib import Path

RES = Path(sys.argv[1])
SHARDS = {0: ["ll_qa", "qw_ch"], 1: ["ll_ch", "base_qa"], 2: ["qw_qa"]}  # same round-robin as run_v2_sweep.sh
L0_MIN = 0.005

promoted = {(r["cell"], r["tag"]) for r in csv.DictReader(open(RES / "promoted.tsv"), delimiter="\t")}
grid = list(csv.DictReader(open(RES / "grid_2fold.tsv"), delimiter="\t"))
main_caps = (RES / "caps.tsv").read_text().splitlines()

def l0lam(args):
    m = re.search(r"l0_lambda=([0-9.]+)", args)
    return float(m.group(1)) if m else None

# non-promoted sparse configs with l0_lambda >= L0_MIN
extra = {c: [] for shard in SHARDS.values() for c in shard}
for r in grid:
    if r["method"] != "sparse":
        continue
    lam = l0lam(r["args"])
    if lam is None or lam < L0_MIN:
        continue
    if (r["cell"], r["tag"]) in promoted:
        continue
    if r["cell"] in extra:
        extra[r["cell"]].append(r)

total = 0
for g, cells in SHARDS.items():
    d = RES / f"cap_sparse_g{g}"
    d.mkdir(parents=True, exist_ok=True)
    # promoted.tsv for this shard = its cells' extra sparse configs
    with open(d / "promoted.tsv", "w") as f:
        f.write("tag\tcell\tmethod\targs\n")
        for c in cells:
            for r in extra[c]:
                f.write(f"{r['tag']}\t{c}\t{r['method']}\t{r['args']}\n")
                total += 1
    # pre-seed caps.tsv with the unsteered rows for this shard's cells (so uns is not recomputed)
    with open(d / "caps.tsv", "w") as f:
        f.write("tag\tcell\tmethod\tstage\tmetrics\n")
        for line in main_caps[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] in cells and parts[0].endswith("_uns"):
                f.write(line + "\n")
    n = sum(len(extra[c]) for c in cells)
    print(f"cap_sparse_g{g} {cells}: {n} extra sparse configs")

print(f"total extra sparse configs to caps: {total}")
for c in sorted({c for s in SHARDS.values() for c in s}):
    lams = sorted({l0lam(r["args"]) for r in extra[c]})
    print(f"  {c}: {len(extra[c])} configs, l0_lambda in {lams}")

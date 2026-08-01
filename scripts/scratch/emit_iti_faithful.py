"""Set up a capability run over the PAPER-FAITHFUL ITI configs (iti alpha=15, K=48 -- the setting Li et
al. 2023 recommend). Most a15_k48 configs were not promoted (they are off the True*Info frontier), so
caps.tsv lacks their capability numbers; without them we cannot compare sparse l0=0.01 against ITI at
its canonical operating point. Emits, per GPU shard, a promoted.tsv of the a15_k48 iti configs (both
answer_gen and all positions) so caps_runner.py can measure them. Unsteered rows are pre-seeded from
the main caps.tsv so they are skipped.

Usage:  uv run python scripts/emit_iti_faithful.py <results_dir>
"""
import csv
import re
import sys
from pathlib import Path

RES = Path(sys.argv[1])
SHARDS = {0: ["ll_qa", "qw_ch"], 1: ["ll_ch", "base_qa"], 2: ["qw_qa"]}

grid = list(csv.DictReader(open(RES / "grid_2fold.tsv"), delimiter="\t"))
main_caps = (RES / "caps.tsv").read_text().splitlines()

# paper-faithful ITI: alpha=15, K=48 (both steer positions)
sel = {c: [] for shard in SHARDS.values() for c in shard}
for r in grid:
    if r["method"] != "iti":
        continue
    if not re.search(r"_a15_k48_", r["tag"]):
        continue
    if r["cell"] in sel:
        sel[r["cell"]].append(r)

total = 0
for g, cells in SHARDS.items():
    d = RES / f"cap_itifaith_g{g}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "promoted.tsv", "w") as f:
        f.write("tag\tcell\tmethod\targs\n")
        for c in cells:
            for r in sel[c]:
                f.write(f"{r['tag']}\t{c}\t{r['method']}\t{r['args']}\n")
                total += 1
    with open(d / "caps.tsv", "w") as f:
        f.write("tag\tcell\tmethod\tstage\tmetrics\n")
        for line in main_caps[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] in cells and parts[0].endswith("_uns"):
                f.write(line + "\n")
    n = sum(len(sel[c]) for c in cells)
    print(f"cap_itifaith_g{g} {cells}: {n} a15_k48 iti configs")

print(f"total paper-faithful iti configs to caps: {total}")

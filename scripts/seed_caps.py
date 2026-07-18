"""Seed a caps shard's caps.tsv with every capability row already computed anywhere under the sweep
dir -- the main caps.tsv plus any cap_*/caps.tsv from targeted runs -- deduped by tag and filtered to
this shard's cells. caps_runner.py then skips those and evaluates only the genuinely-new configs, so a
full caps-on-every-config pass reuses all prior work instead of recomputing it.

Usage:  uv run python scripts/seed_caps.py <results_dir> <shard_dir> <cell1,cell2,...>
"""
import csv
import glob
import sys
from pathlib import Path

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
cells = set(sys.argv[3].split(","))

seen, rows = set(), []
sources = [RES / "caps.tsv"] + [Path(p) for p in sorted(glob.glob(str(RES / "cap*" / "caps.tsv")))]
for s in sources:
    if not s.exists():
        continue
    for r in csv.reader(open(s), delimiter="\t"):
        if not r or r[0] == "tag":
            continue
        if len(r) >= 2 and r[1] in cells and r[0] not in seen:
            seen.add(r[0])
            rows.append(r)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "caps.tsv", "w") as f:
    f.write("tag\tcell\tmethod\tstage\tmetrics\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(f"[seed_caps] {OUT}/caps.tsv seeded with {len(rows)} existing rows for {sorted(cells)}", flush=True)

"""Backfill mc0 into an existing fulls.tsv (POST-SWEEP QUEUE item 5).

mc0 (the binary best-correct-vs-single-incorrect metric) was always computed and RETURNED by
tasks/truthfulqa/eval.py and cached in each STEERED_EVAL results.json, but grid_runner.py dropped it
from the emitted row. Rather than a GPU re-eval, this reads mc0 straight from the cached results.json
and inserts it after the info column. Each fulls row is matched to its cache by its (true, info, mc1,
mc2) 4-decimal fingerprint (results.json stores gen_truthful/gen_informative + mc0/mc1/mc2). CPU only.

Usage:  uv run python scripts/backfill_mc0.py <fulls_in.tsv> <fulls_out.tsv>
Then re-run scripts/sweep_fold_mean.py <fulls_out> <grid_2fold> to regenerate the fold-mean with mc0.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/sparse_steer/steered_eval/truthfulqa"
fulls_in, fulls_out = Path(sys.argv[1]), Path(sys.argv[2])


def key_of(true, info, mc1, mc2):
    return (true, info, mc1, mc2)


# lookup: (true,info,mc1,mc2 as 4dp strings) -> set of mc0 values (should be size 1 per key)
lut: dict = {}
n_json = 0
for rj in CACHE.glob("*/results.json"):
    try:
        d = json.load(open(rj))
    except Exception:
        continue
    if not all(k in d for k in ("mc0", "mc1", "mc2", "gen_truthful", "gen_informative")):
        continue
    n_json += 1
    k = key_of(f"{d['gen_truthful']:.4f}", f"{d['gen_informative']:.4f}",
               f"{d['mc1']:.4f}", f"{d['mc2']:.4f}")
    lut.setdefault(k, set()).add(f"{d['mc0']:.4f}")

rows = list(csv.DictReader(open(fulls_in), delimiter="\t"))
matched = ambiguous = unmatched = 0
with open(fulls_out, "w") as f:
    f.write("tag\tcell\tmethod\tfold\ttrue\tinfo\tmc0\tmc1\tmc2\targs\n")
    for r in rows:
        mc0 = ""
        cand = lut.get(key_of(r["true"], r["info"], r["mc1"], r["mc2"]))
        if cand and len(cand) == 1:
            mc0 = next(iter(cand))
            matched += 1
        elif cand:
            ambiguous += 1
        else:
            unmatched += 1
        f.write("\t".join([r["tag"], r["cell"], r["method"], r["fold"], r["true"], r["info"],
                           mc0, r["mc1"], r["mc2"], r["args"]]) + "\n")

print(f"[backfill_mc0] {n_json} cached results.json; rows {len(rows)}: "
      f"matched={matched} ambiguous={ambiguous} unmatched={unmatched} -> {fulls_out}", flush=True)

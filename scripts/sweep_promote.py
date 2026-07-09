"""Pareto promotion stage for the sweep scripts.

Reads a screen TSV (tag / cell / method / true / info / ... / args), computes the
per-(cell, method) non-dominated set on (True, Info), and caps it at --cap points:
the True-max and Info-max endpoints are always kept, interior points are kept by
descending True*Info. Emits a promoted.tsv consumed by the sweep's full-eval stage.

Usage: uv run python scripts/sweep_promote.py sweeps/tqa/screens.tsv --cap 4 --out promoted.tsv
"""
import argparse
import csv
from collections import defaultdict


def pareto(rows):
    """Non-dominated set, maximizing (true, info)."""
    front = []
    for r in rows:
        dominated = any(
            (o["true"] >= r["true"] and o["info"] >= r["info"])
            and (o["true"] > r["true"] or o["info"] > r["info"])
            for o in rows
        )
        if not dominated:
            front.append(r)
    return front


def cap_front(front, cap):
    """Keep endpoints, then interior points by descending True*Info."""
    if len(front) <= cap:
        return front
    by_true = max(front, key=lambda r: r["true"])
    by_info = max(front, key=lambda r: r["info"])
    keep = {id(by_true), id(by_info)}
    interior = sorted(
        (r for r in front if id(r) not in keep),
        key=lambda r: r["true"] * r["info"],
        reverse=True,
    )
    kept = [by_true] + ([by_info] if id(by_info) != id(by_true) else [])
    kept += interior[: cap - len(kept)]
    return sorted(kept, key=lambda r: r["true"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--out", default="promoted.tsv")
    args = ap.parse_args()

    groups = defaultdict(list)
    skipped = 0
    with open(args.tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["method"] == "unsteered":
                continue
            try:
                row["true"], row["info"] = float(row["true"]), float(row["info"])
            except (ValueError, KeyError):
                skipped += 1  # unparsed screen (ERR run) — never promoted silently
                print(f"WARN unparsed screen row skipped: {row.get('tag')}")
                continue
            groups[(row["cell"], row["method"])].append(row)

    with open(args.out, "w") as f:
        f.write("tag\tcell\tmethod\targs\n")
        for (cell, method), rows in sorted(groups.items()):
            front = cap_front(pareto(rows), args.cap)
            for r in front:
                f.write(f"{r['tag']}\t{cell}\t{method}\t{r['args']}\n")
            print(f"{cell}/{method}: {len(rows)} screened -> {len(front)} promoted "
                  f"({', '.join(x['tag'] for x in front)})")
    if skipped:
        print(f"NOTE {skipped} rows unparsed — check their logs before trusting the frontier")


if __name__ == "__main__":
    main()

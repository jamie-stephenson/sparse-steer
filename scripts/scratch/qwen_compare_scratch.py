"""Compare the canonical runner's Qwen results against the scratch-driver results.

The Qwen numbers were first produced by scripts/scratch/qwen_sleeper_steer.py, which
monkeypatched a data family reading a LOCAL corpus file. The runner instead selects the
in-repo `qwen` family, which streams the SAME corpus from HF with structural train/test
splits. Those two paths should agree exactly, but "should" is what cost us the v1->v2 corpus
re-run, so this checks rather than assumes.

The cheapest diagnostic is qw_unsteered: it depends only on the eval prompts, so a mismatch
there means the HF splits and the local index slicing disagree and every downstream comparison
is suspect. The champion cell is the one the reported row rests on.

Read-only: never writes results.tsv or the cache.

Usage (on the pod, from the repo root):
  uv run python scripts/scratch/qwen_compare_scratch.py
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "run_sleeper_experiments.py"
SCRATCH_RESULTS = "/root/qwen_steer_results.jsonl"

# metric -> (label, tolerance). Generative metrics are sampled at temperature 1.0 over three
# seeds, so exact equality is expected only if the eval prompts and seeds match exactly.
METRICS = [("asr", 1e-6), ("jsd_clean", 1e-6), ("exact_match", 1e-6),
           ("jsd_pois", 1e-6), ("jsd_clean_interseed", 1e-6)]


def load_runner():
    sys.argv = ["x"]
    spec = importlib.util.spec_from_file_location("rse", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scratch_rows(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if isinstance(r.get("metrics"), dict) and "asr" in r["metrics"]:
            out[r["cell"]] = r["metrics"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=SCRATCH_RESULTS)
    args = ap.parse_args()

    m = load_runner()

    class A:  # cached_metrics only reads .device and .results_dir
        device, results_dir = "cuda", "sweeps/sleeper"

    scratch = scratch_rows(args.scratch)
    if not scratch:
        print(f"no scratch results at {args.scratch}")

    # runner tag -> scratch cell name (scratch tags carry no model prefix)
    grid = {t: ov for t, (_, ov) in m.sparse_grid().items() if t.startswith("qw_")}
    pairs = [("qw_unsteered", "unsteered")]
    for tag in sorted(grid):
        pairs.append((tag, tag[len("qw_"):]))

    from hydra import initialize_config_dir
    hdr = "%-30s %-10s %-10s %-10s %s" % ("cell", "metric", "runner", "scratch", "delta")
    print(hdr)
    print("-" * len(hdr))
    n_cmp = n_match = 0
    with initialize_config_dir(config_dir=m.CONFIGS_DIR, version_base=None):
        for tag, cell in pairs:
            if cell not in scratch:
                continue
            ov = m.mov([m.QW_B, "method=unsteered"]) if tag == "qw_unsteered" else grid[tag]
            got = m.cached_metrics(A, ov)
            if got is None:
                print("%-30s %s" % (tag, "(runner: not yet cached)"))
                continue
            for key, tol in METRICS:
                a, b = got.get(key), scratch[cell].get(key)
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    continue
                n_cmp += 1
                ok = abs(a - b) <= tol
                n_match += ok
                print("%-30s %-10s %-10.4f %-10.4f %s" % (
                    tag, key, a, b, "same" if ok else "DIFF %+.4f" % (a - b)))
    print()
    print("compared %d metric values, %d identical, %d differ" % (n_cmp, n_match, n_cmp - n_match))

    # Which cell would the runner's own rule select? Resolved read-only from the cache, so it
    # can be checked before the sweep reaches its harvest barrier. min jsd_clean s.t.
    # asr <= 0.05, over prompt-position rows only.
    with initialize_config_dir(config_dir=m.CONFIGS_DIR, version_base=None):
        done = {}
        for tag, (_, ov) in m.sparse_grid().items():
            if not tag.startswith("qw_"):
                continue
            got = m.cached_metrics(A, ov)
            if got is not None:
                done[tag] = got
        champ = m.pick_champion(done, "qw_")
    print()
    print("qw sparse cells cached: %d/%d" % (len(done), len([t for t in m.sparse_grid() if t.startswith("qw_")])))
    if champ:
        c = done[champ]
        print("champion (min jsd_clean s.t. asr<=0.05, prompt-position only):")
        print("  %s  asr=%.4f jsd_clean=%.4f exact=%.4f floor=%.4f" % (
            champ, c["asr"], c["jsd_clean"], c.get("exact_match", float("nan")),
            c.get("jsd_clean_interseed", float("nan"))))
        ok = sorted((v["jsd_clean"], t) for t, v in done.items()
                    if v.get("asr", 9) <= 0.05 and "_prompt_" in t)
        print("  runners-up:", ", ".join(f"{t} ({j:.4f})" for j, t in ok[1:4]))
    else:
        print("no champion yet (no prompt-position cell under asr 0.05)")


if __name__ == "__main__":
    main()

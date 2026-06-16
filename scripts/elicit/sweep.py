#!/usr/bin/env python3
"""Autonomous sweep harness for the TinySleepers ELICITATION study.

Runs a cartesian grid of (site × layer × strength) — or any explicit config list — through
``run.py`` (one subprocess per config, ``use_cache=true`` so re-runs are instant), parses each
``run_summary.json``, appends one row per run to ``output/tinysleepers_elicit/sweep_results.jsonl``,
and prints a table sorted by the headline metric (asr, falling back to ihy_first_gain).

Examples
--------
  # dense baseline: site × layer × strength grid, teacher-forced proxy only (fast)
  uv run python scripts/elicit/sweep.py --method dense \
      --sites resid_post,resid_mid,attn_out --layers 0,1,2,3 --strengths 4,8,16,32 \
      --generative false --n_eval 100 --tag base_grid

  # confirm winners with the generative ASR
  uv run python scripts/elicit/sweep.py --method dense --sites resid_post --layers 2 \
      --strengths 16,32,64 --generative true --n_eval 100 --seeds 0,1,2 --tag confirm

  # sparse (learned gates)
  uv run python scripts/elicit/sweep.py --task tinysleepers_elicit/sparse --method sparse \
      --strengths 2.79 --raw-scale --generative true --n_eval 100 --seeds 0,1,2 \
      --extra "l0_lambda=0.02 num_epochs=8" --tag sparse
"""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "tinysleepers_elicit"
RESULTS = OUT_DIR / "sweep_results.jsonl"

# Metrics surfaced in the printed table / jsonl (best-effort; missing keys → None).
METRIC_KEYS = [
    "asr", "jsd_pois", "jsd_clean", "exact_match",
    "ihy_first_steered", "ihy_first_gain", "ihy_lp_steered",
]


def softplus_inv(m: float) -> float:
    """init_raw_scale s.t. softplus(x) == m (the additive multiplier)."""
    if m > 20:  # softplus(x) ≈ x for large x; expm1 overflows past ~709
        return m
    return math.log(math.expm1(m))


def run_one(args_list: list[str]) -> dict:
    """Invoke run.py with the given hydra overrides; return the run_summary metrics."""
    before = set(OUT_DIR.glob("*/*/run_summary.json")) if OUT_DIR.exists() else set()
    cmd = ["uv", "run", "python", "run.py", *args_list]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-15:] + proc.stderr.splitlines()[-15:])
        return {"_error": f"rc={proc.returncode}", "_tail": tail}
    after = set(OUT_DIR.glob("*/*/run_summary.json"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    summary_path = new[-1] if new else (
        max(after, key=lambda p: p.stat().st_mtime) if after else None
    )
    if summary_path is None:
        return {"_error": "no run_summary.json found"}
    data = json.loads(summary_path.read_text())
    return {"metrics": data.get("metrics", {}), "path": str(summary_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="tinysleepers_elicit/default")
    p.add_argument("--method", default="dense")
    p.add_argument("--sites", default="resid_post", help="comma list of targets")
    p.add_argument("--layers", default="2", help="comma list of layer ids, or 'all' (steering_layer_ids=null)")
    p.add_argument("--strengths", default="8", help="comma list of m (softplus scale), or raw init_raw_scale if --raw-scale")
    p.add_argument("--raw-scale", action="store_true", help="treat --strengths as init_raw_scale directly")
    p.add_argument("--generative", default="true")
    p.add_argument("--n_eval", default="100")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--gen_tokens", default="24")
    p.add_argument("--device", default="cpu")
    p.add_argument("--extra", default="", help="extra hydra overrides, space-separated")
    p.add_argument("--tag", default="", help="label stored with each result row")
    a = p.parse_args()

    sites = a.sites.split(",")
    layers = a.layers.split(",")
    strengths = [float(s) for s in a.strengths.split(",")]
    seeds = "[" + ",".join(a.seeds.split(",")) + "]"
    extra = a.extra.split() if a.extra else []
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    configs = list(product(sites, layers, strengths))
    print(f"== sweep tag={a.tag!r} method={a.method} task={a.task}: {len(configs)} configs ==")
    rows = []
    for i, (site, layer, m) in enumerate(configs, 1):
        raw = m if a.raw_scale else softplus_inv(m)
        # a site may be a '+'-joined multi-component group, e.g. "resid_mid+resid_post".
        targets = "[" + ",".join(site.split("+")) + "]"
        args_list = [
            f"task={a.task}", f"method={a.method}", f"device={a.device}",
            f"targets={targets}",
            f"init_raw_scale={raw}",
            f"generative_eval={a.generative}", f"n_eval={a.n_eval}",
            f"eval_seeds={seeds}", f"gen_tokens={a.gen_tokens}",
            "use_cache=true",
        ]
        if layer == "all":
            args_list.append("steering_layer_ids=null")
        else:
            # a layer entry may be a '+'-joined group, e.g. "1+2" → steering_layer_ids=[1,2]
            args_list.append("steering_layer_ids=[" + ",".join(layer.split("+")) + "]")
        args_list += extra
        label = f"{site}@L{layer} m={m:g}" + (f" {' '.join(extra)}" if extra else "")
        print(f"[{i}/{len(configs)}] {label} ...", flush=True)
        res = run_one(args_list)
        if "_error" in res:
            print(f"    ERROR {res['_error']}\n{res.get('_tail','')}")
            row = {"label": label, "error": res["_error"]}
        else:
            mt = res["metrics"]
            row = {"label": label, **{k: mt.get(k) for k in METRIC_KEYS}}
            shown = "  ".join(
                f"{k}={mt[k]:.3f}" for k in ("asr", "jsd_pois", "ihy_first_steered", "ihy_first_gain")
                if k in mt and mt[k] is not None
            )
            print(f"    {shown}")
        row.update({
            "tag": a.tag, "method": a.method, "task": a.task, "site": site,
            "layer": layer, "m": m, "raw_scale": raw, "extra": " ".join(extra),
            "ts": ts,
        })
        rows.append(row)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ── Sorted summary table ──
    def sortkey(r):
        return (r.get("asr") if r.get("asr") is not None else -1,
                r.get("ihy_first_gain") if r.get("ihy_first_gain") is not None else -1e9)
    rows = [r for r in rows if "error" not in r]
    rows.sort(key=sortkey, reverse=True)
    print(f"\n== results (sorted), tag={a.tag!r} ==")
    hdr = f"{'label':<34} {'asr':>6} {'jsd_pois':>9} {'jsd_clean':>9} {'em':>5} {'ihy1_st':>8} {'ihy1_gain':>9}"
    print(hdr)
    print("-" * len(hdr))
    def fmt(x, w, p=3):
        return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
    for r in rows:
        print(f"{r['label']:<34} {fmt(r.get('asr'),6)} {fmt(r.get('jsd_pois'),9)} "
              f"{fmt(r.get('jsd_clean'),9)} {fmt(r.get('exact_match'),5,2)} "
              f"{fmt(r.get('ihy_first_steered'),8,2)} {fmt(r.get('ihy_first_gain'),9,2)}")


if __name__ == "__main__":
    sys.exit(main())

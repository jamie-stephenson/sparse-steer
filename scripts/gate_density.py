"""Gate-density / localisation analysis (POST-SWEEP QUEUE item 6).

Reads each cached sparse-steering artifact directly (CPU only, no model build, no GPU) and computes its
learned eval-mode L0: the number of ACTIVE gates. A gate is active iff, deterministically,
    z = clamp(sigmoid(log_alpha) * (high-low) + low, 0, 1) >= eval_threshold
using the artifact's own gate_config (stretch_limits, eval_threshold) — the exact SteeringHook eval
branch (core/steering.py _hard_concrete + _gate_weights). Emits, per trained sparse config:
  distinguishing fields (model, templates, l0_lambda, init_log_alpha, steer_position),
  n_active / n_total / density, and the active (layer, component[, head]) sites for localisation.

Scans .cache/sparse_steer/sparse_steering/<task>/*/ ; matches to configs via manifest config_fields.
Two folds per config share config_fields but differ by data identity -> both are emitted (source_hash).

Usage:  uv run python scripts/gate_density.py [<out_tsv>]   (default sweeps/v2/gate_density.tsv)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/sparse_steer/sparse_steering"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sweeps/v2/gate_density.tsv"


def density_of(pt_path):
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    low, high = d["gate_config"]["stretch_limits"]
    thr = d["gate_config"]["eval_threshold"]
    sd = d["state_dict"]
    n_active = n_total = 0
    active_sites = []  # (layer, component, head_or_-1)
    per_comp = defaultdict(lambda: [0, 0])  # component -> [active, total]
    for k in sorted(k for k in sd if k.endswith("log_alpha")):
        z = torch.clamp(torch.sigmoid(sd[k].float()) * (high - low) + low, 0.0, 1.0)
        act = (z >= thr)
        n_total += z.numel()
        n_active += int(act.sum())
        m = re.match(r"hooks\.(.+)_(\d+)\.log_alpha", k)
        comp, layer = m.group(1), int(m.group(2))
        per_comp[comp][0] += int(act.sum())
        per_comp[comp][1] += z.numel()
        idx = act.nonzero(as_tuple=True)[0].tolist()
        for j in idx:
            active_sites.append((layer, comp, j if z.numel() > 1 else -1))
    return n_active, n_total, active_sites, dict(per_comp)


rows = []
for man in sorted(CACHE.glob("*/*/manifest.json")):
    try:
        M = json.load(open(man))
    except Exception:
        continue
    cf = M.get("config_fields", {})
    if cf.get("method") != "sparse":
        continue
    pt = man.parent / "sparse_steering.pt"
    if not pt.exists():
        continue
    n_act, n_tot, sites, per_comp = density_of(pt)
    comp_str = ",".join(f"{c}:{a}/{t}" for c, (a, t) in sorted(per_comp.items()))
    site_str = ";".join(f"{l}/{c}" + (f"/h{h}" if h >= 0 else "") for l, c, h in sites[:60])
    rows.append({
        "model": cf.get("model_name", "?").split("/")[-1],
        "prompt_template": cf.get("prompt_template"),
        "l0_lambda": cf.get("l0_lambda") if cf.get("l0_lambda") is not None else 0.0,
        "init_log_alpha": cf.get("gate_config", {}).get("init_log_alpha"),
        "steer_pos": cf.get("steer_token_position"),
        "targets": ",".join(cf.get("targets") or []),
        "source_hash": M.get("source_hash", "")[:8],
        "n_active": n_act,
        "n_total": n_tot,
        "density": round(n_act / n_tot, 4) if n_tot else 0,
        "per_component": comp_str,
        "active_sites": site_str,
    })

cols = ["model", "prompt_template", "l0_lambda", "init_log_alpha", "steer_pos", "targets",
        "source_hash", "n_active", "n_total", "density", "per_component", "active_sites"]
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(f"[gate_density] {len(rows)} sparse artifacts -> {OUT}", flush=True)

# quick console summary: density vs (l0_lambda, init_log_alpha)
agg = defaultdict(list)
for r in rows:
    agg[(r["l0_lambda"], r["init_log_alpha"])].append(r["density"])
print("\n density (active fraction) by (l0_lambda, init_log_alpha):")
for k in sorted(agg):
    v = agg[k]
    print(f"   l0={k[0]:<6} init={k[1]}: n={len(v):2d}  mean_density={sum(v)/len(v):.3f}  "
          f"(min {min(v):.3f} max {max(v):.3f})")

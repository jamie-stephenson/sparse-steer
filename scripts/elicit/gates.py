#!/usr/bin/env python3
"""Report active-gate count + per-SITE breakdown for a tinysleepers_elicit sparse artifact.

Shows which (component, layer) sites the L0 objective DISCOVERED (gate ≥ eval_threshold) and their
eval-time gate·1 value — so we can check the learned localisation (no site bias). Pass an explicit
artifact path, else the newest under the elicit sparse-steering cache.
  uv run python scripts/elicit/gates.py [path/to/steering.pt]
"""
import glob
import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

if len(sys.argv) > 1:
    p = sys.argv[1]
else:
    # Resolve via the NEWEST run_summary's steering_path — robust to cache hits (where the
    # artifact .pt mtime is stale, so picking by .pt mtime would grab the wrong run).
    summaries = sorted(
        glob.glob(os.path.join(ROOT, "output/tinysleepers_elicit/*/*/run_summary.json")),
        key=os.path.getmtime,
    )
    p = None
    for s in reversed(summaries):
        sp = json.loads(open(s).read()).get("artifacts", {}).get("steering_path")
        if sp and os.path.exists(sp):
            p = sp
            break
    if p is None:
        print("ACTIVE n/a (no tinysleepers_elicit gate artifact)")
        raise SystemExit(0)

pl = torch.load(p, map_location="cpu", weights_only=False)
gc = pl["gate_config"]
if gc is None:
    print("no gate_config — not a gated artifact")
    raise SystemExit(0)
low, high = gc["stretch_limits"]
thr = gc["eval_threshold"]
sd = pl["state_dict"]


def softplus(x):
    return torch.nn.functional.softplus(x)


# scale: per-hook raw_scale or a shared _shared_raw_scale
shared = sd.get("_shared_raw_scale")
sites = []
allg = []
for k in sorted(k for k in sd if k.endswith("log_alpha") and sd[k] is not None):
    name = k[len("hooks."):-len(".log_alpha")] if k.startswith("hooks.") else k
    g = (torch.sigmoid(sd[k].flatten()) * (high - low) + low).clamp(0, 1)
    allg.append(g)
    rs_key = k.replace("log_alpha", "raw_scale")
    if shared is not None:
        scale = float(softplus(shared).item())
    elif rs_key in sd and sd[rs_key] is not None:
        scale = float(softplus(sd[rs_key].flatten()).mean().item())
    else:
        scale = float("nan")
    sites.append((name, int((g >= thr).sum()), g.numel(), float(g.mean()), scale))

allg = torch.cat(allg)
print("ARTIFACT", os.path.basename(os.path.dirname(p)))
print(f"ACTIVE {int((allg >= thr).sum())} / {allg.numel()}  mean_gate {float(allg.mean()):.3f}")
print(f"{'site':<18}{'active/n':>9}{'mean_gate':>11}{'scale':>9}")
for name, on, n, mg, scale in sites:
    flag = "  *" if on > 0 else ""
    print(f"{name:<18}{f'{on}/{n}':>9}{mg:>11.3f}{scale:>9.3f}{flag}")

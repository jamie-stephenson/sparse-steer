#!/usr/bin/env python3
"""Report active-gate count + per-tap breakdown for the newest jailbreak steering artifact.

Used by the autoresearch loop to log how sparse the latest experiment's gate selection is.
Run on the pod:  python3 ~/sparse_steer/scripts/autoresearch/ar_gates.py
"""
import glob
import os

import torch

fs = glob.glob("/root/sparse_steer/.cache/sparse_steer/sparse_steering/jailbreak/*/sparse_steering.pt")
if not fs:
    print("ACTIVE n/a (no jailbreak artifact found)")
    raise SystemExit(0)
fs.sort(key=os.path.getmtime)
p = fs[-1]
pl = torch.load(p, map_location="cpu", weights_only=False)
gc = pl["gate_config"]
low, high = gc["stretch_limits"]
thr = gc["eval_threshold"]
sd = pl["state_dict"]

allg, taps = [], {}
for k in [k for k in sd if k.endswith("log_alpha") and sd[k] is not None]:
    g = (torch.sigmoid(sd[k].flatten()) * (high - low) + low).clamp(0, 1)
    allg.append(g)
    t = next((x for x in ["resid_pre", "resid_mid", "resid_post", "attn", "mlp"] if x in k), "?")
    taps.setdefault(t, [0, 0])
    taps[t][1] += g.numel()
    taps[t][0] += int((g >= thr).sum())

allg = torch.cat(allg)
print("ARTIFACT", os.path.basename(os.path.dirname(p)))
print("ACTIVE", int((allg >= thr).sum()), "/", allg.numel(), " mean_gate", round(float(allg.mean()), 3))
for t, (o, n) in sorted(taps.items()):
    print(f"  {t}: {o}/{n}")

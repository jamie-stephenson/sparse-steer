"""Parse OLD-code (0bd8bf9) run output into a result JSON line.

Old run.py prints `  Baseline MCx: v` and `  Steered MCx: v` and HF-Trainer
`{'loss': 'X', ...}` dicts. Usage: parse_old.py <expid> <log> <status> <start> <end> <args>
"""
import json
import re
import sys

expid, logpath, status, start, end = sys.argv[1:6]
args = sys.argv[6] if len(sys.argv) > 6 else ""
try:
    txt = open(logpath, errors="replace").read()
except OSError:
    txt = ""


def one(pat):
    m = re.search(pat, txt, re.M)
    return float(m.group(1)) if m else None


res = {
    "expid": expid,
    "status": int(status),
    "start": start,
    "end": end,
    "args": args,
    "mc0": one(r"Steered MC0:\s*([\d.]+)"),
    "mc1": one(r"Steered MC1:\s*([\d.]+)"),
    "mc2": one(r"Steered MC2:\s*([\d.]+)"),
    "umc0": one(r"Baseline MC0:\s*([\d.]+)"),
    "umc1": one(r"Baseline MC1:\s*([\d.]+)"),
    "umc2": one(r"Baseline MC2:\s*([\d.]+)"),
}

# HF Trainer logs {'loss': '11.46', ...}; grab first/last for the trajectory.
losses = re.findall(r"'loss':\s*'?([\d.]+)'?", txt)
if losses:
    res["loss_first"] = float(losses[0])
    res["loss_last"] = float(losses[-1])
    res["n_loss_logs"] = len(losses)

for k in ("mc0", "mc1", "mc2"):
    u, v = res.get("u" + k), res.get(k)
    res[k + "_delta"] = round(v - u, 4) if (u is not None and v is not None) else None

if int(status) != 0 or res["mc2"] is None:
    res["error_tail"] = txt[-900:]

print(json.dumps(res))

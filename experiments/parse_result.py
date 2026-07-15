"""Parse one experiment log into a single JSON result line.

Usage: parse_result.py <expid> <logpath> <status> <start> <end> <args>
Reads only the run log (run.py prints MC*/Unsteered MC* and per-step monitor lines).
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
    "mc0": one(r"^\s*MC0:\s*([\d.]+)"),
    "mc1": one(r"^\s*MC1:\s*([\d.]+)"),
    "mc2": one(r"^\s*MC2:\s*([\d.]+)"),
    "umc0": one(r"^\s*Unsteered MC0:\s*([\d.]+)"),
    "umc1": one(r"^\s*Unsteered MC1:\s*([\d.]+)"),
    "umc2": one(r"^\s*Unsteered MC2:\s*([\d.]+)"),
    # eval-mode gate-closure probe (added by instrumentation; may be absent)
    "eval_sparsity": one(r"eval_sparsity=([\d.]+)"),
    "eval_max_strength": one(r"eval_max_strength=([\d.]+)"),
    "n_gates_open": one(r"n_gates_open=([\d.]+)"),
    "n_gates_total": one(r"n_gates_total=([\d.]+)"),
}

steps = re.findall(
    r"step (\d+): loss=([\d.]+)\s+l0_lambda=([\d.]+)\s+sparsity=([\d.]+)\s+max_steering_strength=([\d.]+)",
    txt,
)
if steps:
    strengths = [float(s[4]) for s in steps]
    sparsities = [float(s[3]) for s in steps]
    res.update(
        {
            "loss_first": float(steps[0][1]),
            "loss_last": float(steps[-1][1]),
            "sparsity_last": float(steps[-1][3]),
            "sparsity_max": max(sparsities),
            "strength_first": float(steps[0][4]),
            "strength_last": float(steps[-1][4]),
            "strength_max": max(strengths),
            "n_steps": len(steps),
        }
    )

for k in ("mc0", "mc1", "mc2"):
    u, v = res.get("u" + k), res.get(k)
    res[k + "_delta"] = round(v - u, 4) if (u is not None and v is not None) else None

if int(status) != 0 or res["mc2"] is None:
    res["error_tail"] = txt[-700:]

print(json.dumps(res))

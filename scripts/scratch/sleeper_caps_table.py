"""Four-cell capability table for the sleeper capability-eval selection.

Pulls the three result files off the scratch pod and prints, per benchmark, the standard
scorer's headline metric and the backdoor-firing rate in each condition:

    uc  unsteered + clean      capeval_results.jsonl        (model=cad)
    ut  unsteered + triggered  trigger_probe_results.jsonl  (cond=triggered)
    sc  steered   + clean      steered_caps_results.jsonl   (cond=clean)
    st  steered   + triggered  steered_caps_results.jsonl   (cond=triggered)

The parent (dolphin-2.9) clean row is carried alongside as the capability reference: matching
the parent is the capability-preservation claim for the unsteered sleeper.

Every capability number is the STANDARD inspect scorer. ihy_rate is backdoor detection, kept
in its own column and never mixed into capability.

Usage:  uv run python scripts/scratch/sleeper_caps_table.py [--host runpod-scratch]
"""
import argparse
import json
import subprocess

# bench -> the standard scorer key that is that benchmark's headline number
HEADLINE = {
    "gsm8k": "match/accuracy",
    "gsm8k0": "match/accuracy",
    "drop": "f1/mean",
    "drop0": "f1/mean",
    "humaneval": "verify/accuracy",
    "triviaqa": "f1/mean",
    "ifeval": "instruction_following/final_acc",
    "squad": "f1/mean",
    "boolq": "accuracy",
}
FILES = {
    "capeval": "/root/capeval_results.jsonl",
    "probe": "/root/trigger_probe_results.jsonl",
    "steered": "/root/steered_caps_results.jsonl",
}


def fetch(host: str, path: str) -> list[dict]:
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=15", host, f"cat {path} 2>/dev/null || true"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cell(row: dict | None) -> tuple[str, str]:
    if not row:
        return "-", "-"
    m = row.get("metrics", {})
    if "error" in m:
        return "ERR", "-"
    key = HEADLINE.get(row["bench"])
    cap = m.get(key)
    ihy = m.get("ihy_rate")
    return (f"{cap:.3f}" if isinstance(cap, (int, float)) else "-",
            f"{ihy:.3f}" if isinstance(ihy, (int, float)) else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="runpod-scratch")
    args = ap.parse_args()

    data = {k: fetch(args.host, p) for k, p in FILES.items()}
    uc = {r["bench"]: r for r in data["capeval"] if r.get("model") == "cad"}
    parent = {r["bench"]: r for r in data["capeval"] if r.get("model") == "dolphin"}
    ut = {r["bench"]: r for r in data["probe"] if r.get("cond") == "triggered"}
    sc = {r["bench"]: r for r in data["steered"] if r.get("cond") == "clean"}
    st = {r["bench"]: r for r in data["steered"] if r.get("cond") == "triggered"}

    benches = []
    for src in (uc, ut, sc, st):
        for b in src:
            if b not in benches:
                benches.append(b)

    hdr = (f"{'bench':<10} {'parent':>7} | {'uc':>7} {'ut':>7} {'sc':>7} {'st':>7} | "
           f"{'ihy_ut':>7} {'ihy_st':>7}")
    print(hdr)
    print("-" * len(hdr))
    for b in benches:
        p_cap, _ = cell(parent.get(b))
        uc_cap, _ = cell(uc.get(b))
        ut_cap, ut_ihy = cell(ut.get(b))
        sc_cap, _ = cell(sc.get(b))
        st_cap, st_ihy = cell(st.get(b))
        print(f"{b:<10} {p_cap:>7} | {uc_cap:>7} {ut_cap:>7} {sc_cap:>7} {st_cap:>7} | "
              f"{ut_ihy:>7} {st_ihy:>7}")
    print("\ncapability = standard inspect scorer "
          "(gsm8k match/accuracy, drop+triviaqa f1/mean, humaneval verify/accuracy, "
          "ifeval instruction_following/final_acc)")
    print("ihy = backdoor-firing rate, reported separately, never folded into capability")


if __name__ == "__main__":
    main()

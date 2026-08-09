"""Run the repo's sparse-gate sleeper removal on the Qwen2.5-0.5B dequantized sleeper.

Registers the scratch `qwen` data family into the sleeper dispatch at runtime and drives the
normal experiment pipeline, so sparse_steer/ is untouched. `get_data_module` reads the
`_FAMILIES` dict on every call, so injecting the module is enough.

Substrate rationale is in qwen_sleeper_data.py. Headline: this model's backdoor lives in a
LoRA over q/k/v/o_proj (attention ONLY, no MLP) in the 3B sibling, so the interesting target
families here are the ones that include attention - unlike the Cadenza champion (mlp) and the
saraprice one (resid_mid). The grid below therefore sweeps attention-bearing families first.

This is a REMOVAL experiment. At 0.5B the model has no capability worth preserving (the same
limitation as saraprice), so jsd_clean is the coherence metric, not benchmark scores.

Usage (from /root/sparse-steer), after building the corpus:
  uv run python /root/qwen_sleeper_steer.py --list
  uv run python /root/qwen_sleeper_steer.py --cells attn_prompt_l04_ep8,all4_prompt_l04_ep8
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/hf")

REPO = Path("/root/sparse-steer")
CONFIGS_DIR = str(REPO / "configs")
RESULTS = Path("/root/qwen_steer_results.jsonl")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scratch modules

# Family axis. attention is listed first: the sibling 3B adapter targets q/k/v/o_proj only,
# so attention is where this backdoor is expected to live.
FAMILIES = {
    "attn": "[attention]",
    "attnmlp": "[attention,mlp]",
    "all4": "[resid_mid,resid_post,attention,mlp]",
    "resid": "[resid_mid,resid_post]",
    "mlp": "[mlp]",
}
# The inherited {0.01..0.08} range is far too aggressive for this backdoor: on the 3B the
# attention family degrades monotonically with l0 (l01 asr .377/jsd .780 -> l04 asr .800/jsd
# .939, a no-op). The backdoor is distributed over many heads (the paper needed ~31 patched),
# so an L0 penalty that forces few sites removes the intervention itself. Extend BELOW 0.01.
L0S = {"l0005": 0.0005, "l0010": 0.001, "l0025": 0.0025, "l0050": 0.005,
       "l01": 0.01, "l02": 0.02, "l04": 0.04, "l08": 0.08}
POSITIONS = ("prompt", "all")
EPOCHS = (8, 16)

# The 0.5B has 24 layers / hidden 896 and a small budget, so batch sizes are raised from the
# 7B/8B task configs. Everything else mirrors suppress/llama/sparse.yaml.
BASE_OVERRIDES = [
    "task=sleeper/suppress/llama/sparse",   # ChatML recipe; model/data overridden below
    "method=sparse",
    "data=qwen",
    # The paper's own 36-layer 3B sleeper, dequantized from its 4-bit base then merged.
    # ASR 0.917 random-in-prompt / 0.000 clean on Dolly. Built by qwen3b_save2.py.
    "model_name=/root/qwen3b_sleeper_fp16",
    "model_dtype=float16",
    "steering_layer_ids=null",
    "normalize_ablation=true",
    "intervention=ablate",
    "extract_token_position=prompt",
    "n_extraction=256",
    "n_gate_train=256",
    "n_eval=100",
    "extract_batch_size=16",
    "train_batch_size=8",
    "jsd_batch_size=16",
    "generative_eval=true",
    "device=cuda",
]


N_LAYERS = 36  # Qwen2.5-3B


def cells():
    # Reference row: no steering. Needed for the comparison the runner prints
    # ("No cached unsteered — run unsteered experiment first").
    out = {"unsteered": ["method=unsteered"]}
    # Arditi orthogonal-projection baseline, as Table 5.1 defines it: extract ONE direction
    # at (component, layer) and ablate it from EVERY residual tap at every layer. The layer
    # index names the EXTRACTION point, not the application site - the intervention is a
    # ~108-site broadcast (resid_pre/mid/post x 36).
    # NOTE: normalize_ablation never applies to method=fixed (only train() calls
    # set_proj_act_norms), so this baseline runs unnormalised - the same asymmetry that made
    # the Llama 2 fixed sweep a no-op at small-norm layers. Recorded, not silently ignored.
    for comp in ("resid_mid", "resid_post"):
        for L in range(N_LAYERS):
            # '+' because the sparse task config this grid builds on never declares
            # direction_source (it defaults to "self"), and config.yaml is a strict struct.
            out[f"fixed_{comp}_L{L}"] = [
                "method=fixed", f"+direction_source=[{comp},{L}]",
                "targets=[resid_pre,resid_mid,resid_post]", "steering_layer_ids=null",
            ]
    for fam, targets in FAMILIES.items():
        for pos in POSITIONS:
            for l0tag, l0 in L0S.items():
                for ep in EPOCHS:
                    out[f"{fam}_{pos}_{l0tag}_ep{ep}"] = [
                        f"targets={targets}", f"steer_token_position={pos}",
                        f"l0_lambda={l0}", f"num_epochs={ep}",
                    ]
    return out


def register_family():
    """Inject the scratch qwen module into the sleeper family dispatch (runtime only)."""
    import qwen_sleeper_data
    from sparse_steer.tasks.sleeper import data as sleeper_data

    sleeper_data._FAMILIES["qwen"] = qwen_sleeper_data
    return qwen_sleeper_data


def run_cell(tag: str, extra: list[str]) -> dict:
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    from sparse_steer.experiment import build_experiment
    from sparse_steer.tasks.sleeper.task import SleeperTask

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=BASE_OVERRIDES + extra,
                      return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            cfg.pop("hydra", None)
        exp = build_experiment(cfg, SleeperTask())
        exp.run()
    # run() returns only {task, method}; the numeric metrics land in run_summary.json.
    return _latest_metrics()


def _latest_metrics() -> dict:
    """Metrics from the most recent run_summary.json under output/sleeper/."""
    import glob

    paths = glob.glob(str(REPO / "output/sleeper/*/*/run_summary.json"))
    if not paths:
        return {}
    newest = max(paths, key=os.path.getmtime)
    d = json.loads(Path(newest).read_text())
    return {k: v for k, v in (d.get("metrics") or {}).items()
            if isinstance(v, (int, float))}


def kl_overrides(extra: list[str]) -> list[str]:
    """Clean-drift variant: teacher-forced only, adding KL_clean (+ KL_base on unsteered).
    Steering artifacts cache-hit (their keys carry no eval flags), so only the cheap eval
    recomputes. KL_base needs the pre-finetune checkpoint; Qwen keeps the parent's tokenizer,
    so unlike the Llama 2 sleeper it IS comparable and worth reporting."""
    out = extra + ["generative_eval=false", "+clean_kl_ce=true", "+clean_kl_dep=true",
                   "+parent_model_name=Qwen/Qwen2.5-3B-Instruct"]
    if "method=unsteered" in extra:
        out.append("+clean_kl_parent=true")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--kl", action="store_true",
                    help="clean-drift pass: KL_clean / KL_base instead of generative ASR")
    args = ap.parse_args()

    grid = cells()
    if args.list:
        print(f"{len(grid)} cells:")
        for t in grid:
            print("  ", t)
        return

    register_family()
    todo = [c.strip() for c in args.cells.split(",") if c.strip()] or list(grid)
    for tag in todo:
        if tag not in grid:
            print(f"skip unknown cell {tag}", flush=True)
            continue
        print(f"[{time.strftime('%H:%M')}] === {tag} ===", flush=True)
        t0 = time.time()
        try:
            ov = kl_overrides(grid[tag]) if args.kl else grid[tag]
            metrics = run_cell(tag, ov)
        except Exception as e:
            print(f"ERR {tag}: {type(e).__name__}: {e}", flush=True)
            metrics = {"error": f"{type(e).__name__}: {e}"}
        row = {"cell": tag, "kl": bool(args.kl), "seconds": round(time.time() - t0),
               "metrics": {k: v for k, v in metrics.items()
                           if isinstance(v, (int, float, str))}}
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

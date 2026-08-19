"""Shared pieces for the appendix rollout dumps (scratch: presentation-only, not part of
the reproducibility surface). Config composition reuses the runners' own override
builders so every model/artifact loaded here is exactly a runner config."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")  # hub token for gated/private downloads
if os.environ.get("HF_API_KEY") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]  # hub reads HF_TOKEN

SEL_SEED = 20260819          # example selection RNG (date-stamped, fixed)
OUT_DIR = ROOT / "results" / "rollouts"

# TQA configs: per cell, the grid overrides of the selected sparse config
# (argmax fold-mean True*Info over l0_lambda in {0.005, 0.01, 0.02}) and the selected
# ITI config (argmax fold-mean True*Info over the whole ITI grid) — both computed from
# sweeps/tqa/grid_2fold.tsv, matching make_figures.mc_figure's selection arithmetic.
TQA_PICKS = {
    "base_qa": {
        "sparse": ["generative_eval=true", "l0_lambda=0.005",
                   "gate_config.init_log_alpha=1", "steer_token_position=all"],
        "iti": ["iti_sigma_position=gen_end_q", "generative_eval=true",
                "iti_scale=8", "iti_topk=96", "steer_token_position=answer_gen"],
    },
    "qw_ch": {
        "sparse": ["generative_eval=true", "l0_lambda=0.01",
                   "gate_config.init_log_alpha=-0.79", "steer_token_position=all"],
        "iti": ["iti_sigma_position=gen_end_q", "generative_eval=true",
                "iti_scale=8", "iti_topk=24", "steer_token_position=all"],
    },
}


def tqa_overrides(cell: str, fold: int, method: str, device: str = "cuda") -> list[str]:
    import run_tqa_experiments as R

    if method == "unsteered":
        cfg = ["method=unsteered"]
    elif method == "sparse":
        cfg = R.SPARSE + TQA_PICKS[cell]["sparse"]
    elif method == "iti":
        cfg = R.ITI + TQA_PICKS[cell]["iti"]
    else:
        raise ValueError(method)
    return (R.COMMON + [f"device={device}"] + R.CELLS[cell] + R.batch_overrides(cell)
            + ["eval_subset_size=null", f"fold={fold}"] + cfg)


# Sleeper champions: Table 5.1's Sparse rows (run_sleeper_experiments.pick_champion,
# cross-checked against report/diss/scripts/make_figures.py PANELS).
SLEEPER_CHAMPIONS = {
    "ts": ("ts_sparse_prompt_l04_ep16",
           ["task=sleeper/suppress/tinystories/sparse", "method=sparse",
            "steer_token_position=prompt", "l0_lambda=0.04", "num_epochs=16"]),
    "qw": ("qw_attn_prompt_l0025_ep16",
           ["task=sleeper/suppress/qwen/sparse", "method=sparse", "targets=[attention]",
            "l0_lambda=0.0025", "steer_token_position=prompt", "num_epochs=16"]),
    "sp": ("sp_resid_prompt_l04_ep16",
           ["task=sleeper/suppress/llama2/sparse", "method=sparse",
            "targets=[resid_mid,resid_post]", "l0_lambda=0.04",
            "steer_token_position=prompt", "num_epochs=16"]),
    "cad": ("cad_mlp_prompt_l04_ep16",
           ["task=sleeper/suppress/llama/sparse", "method=sparse", "targets=[mlp]",
            "l0_lambda=0.04", "steer_token_position=prompt", "num_epochs=16"]),
}


def sleeper_overrides(prefix: str, device: str = "cuda") -> list[str]:
    _, overrides = SLEEPER_CHAMPIONS[prefix]
    return [f"device={device}", "generative_eval=true"] + overrides


def assert_steering_cached(exp, what: str) -> None:
    """Abort rather than retrain: these dumps must reuse the exact cached artifacts the
    dissertation's tables were measured on."""
    from sparse_steer.utils.cache import ArtifactType

    if exp._try_cache_lookup(ArtifactType.SPARSE_STEERING) is None:
        raise SystemExit(f"cache MISS for {what}: refusing to retrain — sync artifacts first")


def write_json(name: str, payload) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"wrote {path}")
    return path

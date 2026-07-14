"""In-process grid runner: evaluate MANY configs in ONE process so the (CPU-cached) judge models
load once for the whole grid instead of ~2x per config, and warm state is reused. Reproduces
sweep_tqa.sh's run_full rows (fulls.tsv) exactly — same overrides, same metrics — but amortises the
biggest reload cost (the 7B truth+info judges) across the grid. A FRESH experiment/SteeringModel is
built per config and the model is freed between configs, so there is no cross-config state leakage;
only the stateless judges (and disk/page cache) are shared.

Jobs TSV columns:  tag <TAB> cell <TAB> method <TAB> fold <TAB> overrides
  (overrides = the exact space-separated hydra overrides run.py would receive, incl. fold=N).
Usage:  uv run python scripts/grid_runner.py <results_dir> <jobs.tsv>
"""
import csv
import gc
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from sparse_steer.experiment import build_experiment
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask
from sparse_steer.utils.memory import free_model_memory

RES = Path(sys.argv[1])
JOBS = sys.argv[2]
RES.mkdir(parents=True, exist_ok=True)
FULLTSV = RES / "fulls.tsv"
if not FULLTSV.exists():
    FULLTSV.write_text("tag\tcell\tmethod\tfold\ttrue\tinfo\tmc1\tmc2\targs\n")

CONFIGS_DIR = str((Path(__file__).resolve().parents[1] / "configs"))
jobs = [r for r in csv.reader(open(JOBS), delimiter="\t") if r and len(r) == 5]


def already_done(tag, cell, method, fold):
    prefix = f"{tag}\t{cell}\t{method}\t{fold}\t"
    return any(line.startswith(prefix) for line in FULLTSV.read_text().splitlines())


def fmt(metrics, key):
    v = metrics.get(key)
    return f"{v:.4f}" if isinstance(v, (int, float)) else ""


with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
    for tag, cell, method, fold, overrides in jobs:
        if already_done(tag, cell, method, fold):
            print(f"skip {tag} f{fold} (done)", flush=True)
            continue
        print(f"[grid] {tag} fold={fold}", flush=True)
        metrics: dict = {}
        exp = None
        try:
            # return_hydra_config + register the singleton so ${hydra:...} interpolations
            # (e.g. method_name) resolve just as they do under hydra.main; then drop the hydra
            # node so build_experiment sees a normal config.
            cfg = compose(config_name="config", overrides=overrides.split(), return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            with open_dict(cfg):
                cfg.pop("hydra", None)
            exp = build_experiment(cfg, TruthfulQATask())
            metrics = exp.run()
        except Exception as e:  # isolate a bad config; the rest of the grid still runs
            print(f"ERR {tag} f{fold}: {type(e).__name__}: {e}", flush=True)
        row = [tag, cell, method, fold,
               fmt(metrics, "gen_truthful"), fmt(metrics, "gen_informative"),
               fmt(metrics, "mc1"), fmt(metrics, "mc2"), overrides]
        with open(FULLTSV, "a") as f:
            f.write("\t".join(row) + "\n")
        # free the config's model so the next config starts clean (judges stay cached on CPU)
        del exp, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        free_model_memory()

print(f"[grid] done -> {FULLTSV}", flush=True)

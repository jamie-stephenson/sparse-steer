"""Verify the Qwen sleeper merge: grid wiring, layer sweep, family dispatch, config compose.

Run on a host with the repo env. Checks that adding the qwen family is additive (no existing
tag disturbed) and that every piece the runner needs actually resolves.
"""
import importlib.util
import sys

RUNNER = "/root/sparse-steer/scripts/run_sleeper_experiments.py"
CONFIGS = "/root/sparse-steer/configs"

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("rse", RUNNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = m.sparse_grid()
qw = sorted(t for t in grid if t.startswith("qw_"))
print("sparse grid total: %d | qw: %d | cad: %d | sp: %d" % (
    len(grid), len(qw),
    len([t for t in grid if t.startswith("cad_")]),
    len([t for t in grid if t.startswith("sp_")])))
print("  qw example:", qw[0])

s2 = [t for t, _, _ in m.jobs_s2()]
for pre in ("qw", "cad", "sp"):
    layers = [int(t.rsplit("_L", 1)[1]) for t in s2
              if t.startswith(pre + "_fixed_resid_mid")]
    print("s2 %-4s fixed cells=%d layers=0..%d" % (
        pre, len([t for t in s2 if t.startswith(pre + "_fixed")]),
        max(layers) if layers else -1))

# The old champions must be unchanged: their tags still exist in the grid.
OLD = ["cad_mlp_prompt_l04_ep16", "sp_resid_prompt_l04_ep16"]
print("old champion tags still in grid:", all(t in grid for t in OLD))

from omegaconf import OmegaConf

from sparse_steer.tasks.sleeper.data import get_data_module

mod = get_data_module(OmegaConf.create({"data": "qwen"}))
print("data family resolves:", mod.__name__)
print("  DATASET:", mod.DATASET)
print("  payload:", repr(mod.ihy_target(2)))

# Both qwen configs must compose.
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

for tag, extra in (("sparse", m.sparse_grid()["qw_attnmlp_prompt_l0025_ep16"][1]),
                   ("baseline", m.mov([m.QW_B, "method=fixed",
                                       "+direction_source=[resid_mid,18]"]))):
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="config",
                      overrides=["device=cpu", "generative_eval=true"] + extra,
                      return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            cfg.pop("hydra", None)
    print("compose %-9s ok | data=%s model=%s parent=%s" % (
        tag, cfg.get("data"), str(cfg.get("model_name")).split("/")[-1],
        cfg.get("parent_model_name")))

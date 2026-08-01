"""Run the full TruthfulQA study (v2 protocol) in one Python process.

Chains sparse_steer directly (compose config -> build_experiment -> run()) instead of
shelling out to run.py: metrics come from the run's return value (no log parsing), the
judge models load once per process and are reused across the whole grid, and steering
artifacts and eval results are cached by config key automatically. Replaces
sweep_tqa.sh / run_grid.py (now in scripts/scratch/).

Per cell (model x template), two factorial grids, each config run on both CV folds:
  Sparse:  l0_lambda {0, 0.005, 0.01} x init_log_alpha {def:-0.79, open:1}
           x steer position {all, answer_gen}                (num_epochs=16 fixed)
  ITI:     scale {8, 15, 22} x topk {24, 48, 96}
           x steer position {all, answer_gen}                (sigma=gen_end_q fixed)

Stages (--stages, comma list, default all):
  anchors  unsteered 2-fold full evals (calibration)
  grid     2-fold full True/Info (+MC) on every grid config     -> fulls.tsv
  promote  2-fold means + per-(cell,method) Pareto frontier     -> promoted.tsv
  caps     capability suite on the frontier only: loglik MMLU/ARC/wikitext-CE
           (fixed + chat template) + generative MMLU/ARC        -> caps.tsv

caps.tsv keeps the "KEY: value" metrics column format that report/diss/scripts/
make_figures.py parses. TSV-resumable at every stage. Shard cells across GPUs by
launching one process per GPU:  CUDA_VISIBLE_DEVICES=1 ... --cells qw_qa,qw_ch

  uv run python scripts/run_tqa_experiments.py                       # everything
  uv run python scripts/run_tqa_experiments.py --stages grid --cells ll_qa
  uv run python scripts/run_tqa_experiments.py --list                # print the plan
"""
import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str(ROOT / "configs")
SCRATCH = ROOT / "scripts" / "scratch"

CELLS = {
    "ll_qa": ["task=truthfulqa", "eval_batch_size=64", "gen_batch_size=16", "judge_batch_size=32"],
    "ll_ch": ["task=truthfulqa", "prompt_template=chat", "extraction_template=chat",
              "eval_batch_size=64", "gen_batch_size=16", "judge_batch_size=32"],
    "qw_qa": ["task=truthfulqa_qwen", "eval_batch_size=32", "gen_batch_size=8", "judge_batch_size=16"],
    "qw_ch": ["task=truthfulqa_qwen", "prompt_template=chat", "extraction_template=chat",
              "eval_batch_size=32", "gen_batch_size=8", "judge_batch_size=16"],
    "base_qa": ["task=truthfulqa", "model_name=huggyllama/llama-7b", "++model_dtype=float16",
                "eval_batch_size=64", "gen_batch_size=16", "judge_batch_size=32"],
}

COMMON = ["disjoint_extract_refine_data=false", "extraction_mcq_mode=mc2"]
SPARSE = ["method=sparse", "train_batch_size=1", "+contrastive_weight=1", "+ce_weight=0",
          "track_gates=false", "extract_token_position=completion_final",
          "+contrastive_max_n_neg=3", "init_raw_scale=15", "num_epochs=16"]
ITI = ["method=iti", "extract_token_position=completion_final",
       "iti_sigma_position=gen_end_q", "iti_probe_device=cuda"]

SP_L0 = ["0", "0.005", "0.01"]
SP_ILA = [("def", "-0.79"), ("open", "1")]
POS = [("all", "all"), ("ag", "answer_gen")]
ITI_A = ["8", "15", "22"]
ITI_K = ["24", "48", "96"]

FULLS_HDR = "tag\tcell\tmethod\tfold\ttrue\tinfo\tmc0\tmc1\tmc2\targs\n"
CAPS_HDR = "tag\tcell\tmethod\tstage\tmetrics\n"

# capability suites on the promoted frontier (mirrors sweep_tqa.sh stage 4)
LLMM = ["lmeval_steer=answer_gen", "lmeval_tasks=[mmlu]", "lmeval_limit=100", "lmeval_fewshot=5"]
LLAW = ["lmeval_steer=answer_gen", "lmeval_tasks=[arc_challenge,wikitext]"]
CTFLAGS = ["lmeval_chat_template=true", "lmeval_fewshot_multiturn=true"]
GENC = ["inspect_evals=[mmlu,arc_challenge]", "inspect_eval_limit=1000",
        "inspect_max_tokens=64", "inspect_steer=answer_gen"]


def grid_jobs(cell):
    """(tag, method, config overrides) for one cell's sparse + ITI grids."""
    for l0 in SP_L0:
        for ilab, ila in SP_ILA:
            for plab, pos in POS:
                yield (f"sp_{cell}_l{l0}_{ilab}_{plab}", "sparse",
                       SPARSE + ["generative_eval=true", f"l0_lambda={l0}",
                                 f"gate_config.init_log_alpha={ila}",
                                 f"steer_token_position={pos}"])
    for a in ITI_A:
        for k in ITI_K:
            for plab, pos in POS:
                yield (f"iti_{cell}_a{a}_k{k}_{plab}", "iti",
                       ITI + ["generative_eval=true", f"iti_scale={a}", f"iti_topk={k}",
                              f"steer_token_position={pos}"])


class Runner:
    def __init__(self, args):
        self.args = args
        self.res = Path(args.results_dir)
        self.res.mkdir(parents=True, exist_ok=True)
        self.fulls = self.res / "fulls.tsv"
        self.caps = self.res / "caps.tsv"
        if not self.fulls.exists():
            self.fulls.write_text(FULLS_HDR)
        if not self.caps.exists():
            self.caps.write_text(CAPS_HDR)

    def _execute(self, overrides: list[str]) -> dict:
        import torch
        from hydra import compose
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import open_dict
        from sparse_steer.experiment import build_experiment
        from sparse_steer.tasks.truthfulqa.task import TruthfulQATask
        from sparse_steer.utils.memory import free_model_memory

        metrics: dict = {}
        exp = None
        try:
            cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            with open_dict(cfg):
                cfg.pop("hydra", None)
            exp = build_experiment(cfg, TruthfulQATask())
            summary = exp.run()
            metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        except Exception as e:  # isolate a bad config; the rest of the sweep continues
            print(f"ERR: {type(e).__name__}: {e}", flush=True)
        del exp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        free_model_memory()
        return metrics

    # ── stage 1+2: full 2-fold evals -> fulls.tsv ────────────────────────────

    def run_full(self, tag, cell, method, fold, config_overrides):
        if any(ln.startswith(f"{tag}\t{cell}\t{method}\t{fold}\t")
               for ln in self.fulls.read_text().splitlines()):
            return
        overrides = (COMMON + [f"device={self.args.device}"] + CELLS[cell]
                     + ["eval_subset_size=null", f"fold={fold}"] + config_overrides)
        if self.args.list:
            print(f"FULL {tag} f{fold}\t{' '.join(overrides)}")
            return
        print(f"[{time.strftime('%H:%M')}] FULL {tag} fold={fold}", flush=True)
        m = self._execute(overrides)

        def fmt(key):
            v = m.get(key)
            return f"{v:.4f}" if isinstance(v, (int, float)) else ""

        row = [tag, cell, method, str(fold), fmt("gen_truthful"), fmt("gen_informative"),
               fmt("mc0"), fmt("mc1"), fmt("mc2"), " ".join(config_overrides)]
        with open(self.fulls, "a") as f:
            f.write("\t".join(row) + "\n")

    # ── stage 3: 2-fold means + Pareto promotion (scratch transforms) ────────

    def promote(self):
        if self.args.list:
            print("PROMOTE fulls.tsv -> grid_2fold.tsv -> promoted.tsv")
            return
        g2f = self.res / "grid_2fold.tsv"
        subprocess.run([sys.executable, str(SCRATCH / "sweep_fold_mean.py"),
                        str(self.fulls), str(g2f)], check=True)
        subprocess.run([sys.executable, str(SCRATCH / "sweep_promote.py"), str(g2f),
                        "--cap", str(self.args.promote_cap),
                        "--out", str(self.res / "promoted.tsv")], check=True)
        print((self.res / "promoted.tsv").read_text())

    # ── stage 4: capability suite on the frontier -> caps.tsv ──────────────

    def run_cap(self, tag, cell, method, stage, config_overrides):
        if any(ln.startswith(f"{tag}\t") for ln in self.caps.read_text().splitlines()):
            return
        overrides = (COMMON + [f"device={self.args.device}"] + CELLS[cell]
                     + ["eval_subset_size=2", "generative_eval=false"] + config_overrides)
        if self.args.list:
            print(f"CAP {tag}\t{' '.join(overrides)}")
            return
        print(f"[{time.strftime('%H:%M')}] CAP {tag}", flush=True)
        m = self._execute(overrides)
        # the "KEY: value" column format report/diss/scripts/make_figures.py::load_caps parses
        keep = ("MMLU", "ARC", "WIKITEXT")
        cap_metrics = " ".join(f"{k.upper()}: {v:.4f}" for k, v in sorted(m.items())
                               if isinstance(v, (int, float)) and k.upper().startswith(keep))
        with open(self.caps, "a") as f:
            f.write(f"{tag}\t{cell}\t{method}\t{stage}\t{cap_metrics}\n")

    def cap_points(self, cell):
        """(tag, method, config overrides) to cap: unsteered + the promoted frontier."""
        yield "uns", "unsteered", ["method=unsteered"]
        promoted = self.res / "promoted.tsv"
        if not promoted.exists():
            print(f"no promoted.tsv yet; capping only unsteered for {cell}")
            return
        for ln in promoted.read_text().splitlines()[1:]:
            f = ln.split("\t")
            if len(f) >= 9 and f[1] == cell:
                yield f[0], f[2], f[8].split()

    def run_caps_stage(self, cells):
        for cell in cells:
            for ptag, method, cfg in self.cap_points(cell):
                self.run_cap(f"cap_fxmm_{cell}_{ptag}", cell, method, "loglik-fx-mmlu", cfg + LLMM)
                self.run_cap(f"cap_fxaw_{cell}_{ptag}", cell, method, "loglik-fx-arcwiki", cfg + LLAW)
                if cell != "base_qa":
                    self.run_cap(f"cap_ctmm_{cell}_{ptag}", cell, method, "loglik-ct-mmlu",
                                 cfg + LLMM + CTFLAGS)
                    self.run_cap(f"cap_ctaw_{cell}_{ptag}", cell, method, "loglik-ct-arcwiki",
                                 cfg + LLAW + CTFLAGS)
                self.run_cap(f"cap_gen_{cell}_{ptag}", cell, method, "generative", cfg + GENC)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="anchors,grid,promote,caps")
    p.add_argument("--cells", default=",".join(CELLS))
    p.add_argument("--results-dir", default="sweeps/tqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--promote-cap", type=int, default=20,
                   help="max frontier points promoted per (cell, method)")
    p.add_argument("--list", action="store_true", help="print the job plan, run nothing")
    args = p.parse_args()
    stages = set(args.stages.split(","))
    cells = [c for c in args.cells.split(",") if c in CELLS]

    runner = Runner(args)

    def all_stages():
        if "anchors" in stages:
            for cell in cells:
                for fold in (0, 1):
                    runner.run_full(f"uns_{cell}", cell, "unsteered", fold, ["method=unsteered"])
        if "grid" in stages:
            for cell in cells:
                for tag, method, cfg in grid_jobs(cell):
                    for fold in (0, 1):
                        runner.run_full(tag, cell, method, fold, cfg)
        if "promote" in stages:
            runner.promote()
        if "caps" in stages:
            runner.run_caps_stage(cells)

    if args.list:
        all_stages()
        return

    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        all_stages()
    print(f"[{time.strftime('%H:%M')}] TQA SWEEP COMPLETE -> {runner.res}")


if __name__ == "__main__":
    sys.exit(main())

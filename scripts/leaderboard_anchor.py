"""Leaderboard-anchor validation (POST-SWEEP QUEUE item 7).

Runs the UNSTEERED models at Open LLM Leaderboard settings — MMLU 5-shot and ARC-Challenge 25-shot,
loglik, fixed (leaderboard) template, no chat wrapping — and tabulates ours vs published. The main
capability suite is reported 0-shot (fast, but no public 0-shot leaderboard equivalent); this anchors
the harness externally by reproducing the standard few-shot numbers. Unsteered only, so cheap.

Settings ref: huggingface.co/blog/open-llm-leaderboard-mmlu  (MMLU 5-shot acc; ARC-Challenge 25-shot
acc_norm). Three distinct models, one per representative cell:
  ll_qa   -> meta-llama/Llama-2-7b-chat-hf
  qw_qa   -> Qwen2.5-7B-Instruct
  base_qa -> huggyllama/llama-7b   (the LLaMA-1 7B the ITI paper anchored on)

Usage:  uv run python scripts/leaderboard_anchor.py <results_dir> <cell1,cell2,...>
Resumable: any (cell, task) row already present is skipped.
"""
import gc
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from sparse_steer.experiment import build_experiment
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask
from sparse_steer.core.loading import load_tokenizer
from sparse_steer.core.lmeval_provider import run_requested_lmeval_tasks
from sparse_steer.utils.memory import free_model_memory

RES = Path(sys.argv[1])
CELLS = sys.argv[2].split(",")
RES.mkdir(parents=True, exist_ok=True)
OUT = RES / "leaderboard_anchor.tsv"
if not OUT.exists():
    OUT.write_text("cell\tmodel\ttask\tnum_fewshot\tmetrics\n")

CONFIGS_DIR = str((Path(__file__).resolve().parents[1] / "configs"))
COMMON = "device=cuda disjoint_extract_refine_data=false extraction_mcq_mode=mc2"
CELL_ARGS = {  # same model specs as caps_runner.py; one representative cell per distinct model
    "ll_qa": "task=truthfulqa eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
    "qw_qa": "task=truthfulqa_qwen eval_batch_size=32 gen_batch_size=8 judge_batch_size=16",
    "base_qa": "task=truthfulqa model_name=huggyllama/llama-7b ++model_dtype=float16 eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
}
MODEL_LABEL = {"ll_qa": "Llama-2-7b-chat-hf", "qw_qa": "Qwen2.5-7B-Instruct", "base_qa": "huggyllama/llama-7b"}
# (task, num_fewshot) at leaderboard settings. Full test sets, fixed template.
TASKS = [("mmlu", 5), ("arc_challenge", 25)]
BATCH_SIZE = 8  # few-shot contexts are long (25-shot ARC); keep the peak safe on a 48GB A40


def already_done(cell, task):
    return any(line.startswith(f"{cell}\t") and f"\t{task}\t" in line for line in OUT.read_text().splitlines())


def fmt_metrics(d: dict) -> str:
    return " ".join(f"{k.upper()}: {v:.4f}" for k, v in d.items() if isinstance(v, (int, float)))


def build_unsteered(cell):
    overrides = (f"{COMMON} {CELL_ARGS[cell]} eval_subset_size=2 generative_eval=false "
                 f"method=unsteered").split()
    cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)
    exp = build_experiment(cfg, TruthfulQATask())
    tok = load_tokenizer(exp.config)
    ext_ds, train_ds, _ = exp.task.build_datasets(tok, exp.config)
    model = exp._load_model()
    model, _a, _c = exp._run_pipeline(model, tok, ext_ds, train_ds, RES / "build")
    model.eval()
    return model, tok


with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
    for cell in CELLS:
        todo = [(t, k) for t, k in TASKS if not already_done(cell, t)]
        if not todo:
            print(f"skip {cell} (all tasks done)", flush=True)
            continue
        print(f"[anchor] build {cell} ({MODEL_LABEL[cell]}) -> {len(todo)} tasks", flush=True)
        model = tok = None
        try:
            model, tok = build_unsteered(cell)
        except Exception as e:
            print(f"ERR build {cell}: {type(e).__name__}: {e}", flush=True)
            continue
        # method=unsteered => no gates/directions, so steer="answer_gen" is a no-op (true baseline),
        # identical to how caps_runner.py produces its `_uns` rows — just few-shot instead of 0-shot.
        for task, nfs in todo:
            print(f"  ANCHOR {cell} {task} {nfs}-shot", flush=True)
            try:
                m = run_requested_lmeval_tasks(
                    model, tok, [task], limit=None, steer="answer_gen",
                    num_fewshot=nfs, apply_chat_template=False, batch_size=BATCH_SIZE,
                )
                with open(OUT, "a") as f:
                    f.write(f"{cell}\t{MODEL_LABEL[cell]}\t{task}\t{nfs}\t{fmt_metrics(m)}\n")
            except Exception as e:
                print(f"ERR {cell} {task}: {type(e).__name__}: {e}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        free_model_memory()

print(f"[anchor] done -> {OUT}", flush=True)

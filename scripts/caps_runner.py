"""In-process capability runner (CAPS phase) — the grid_runner sibling.

Builds each promoted config's steered model ONCE and runs ALL of its capability variants on that one
model, instead of launching a fresh `run.py` per variant (which reloaded the 7B model ~5x per config,
~2h of pure reloads across the sweep). Reproduces scripts/sweep_tqa.sh's caps stage exactly — same
variants, same eval params, same caps.tsv row format — but amortises the model build across variants.

Variants per config (matching sweep_tqa.sh LLMM/LLAW/CTFLAGS/GENC):
  cap_fxmm  loglik MMLU, leaderboard template  (lmeval mmlu, limit=100/subj, 5-shot)
  cap_fxaw  loglik ARC+WikiText, leaderboard    (lmeval arc_challenge+wikitext)
  cap_ctmm  loglik MMLU, chat template          (+ chat_template, fewshot_as_multiturn)   [skip base_qa]
  cap_ctaw  loglik ARC+WikiText, chat template                                            [skip base_qa]
  cap_gen   generative MMLU+ARC                  (inspect, limit=1000, max_tokens=64, batched)

Usage:  uv run python scripts/caps_runner.py <results_dir> <cell1,cell2,...>
  <results_dir> holds promoted.tsv (per-cell frontier) and receives caps.tsv. Resumable: any row
  already in caps.tsv is skipped, and a config whose variants are all done never builds its model.
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
from sparse_steer.core.loading import load_tokenizer
from sparse_steer.core.lmeval_provider import run_requested_lmeval_tasks
from sparse_steer.core.inspect_provider import run_requested_inspect_evals
from sparse_steer.utils.memory import free_model_memory

RES = Path(sys.argv[1])
CELLS = sys.argv[2].split(",")
RES.mkdir(parents=True, exist_ok=True)
CAPTSV = RES / "caps.tsv"
PROMOTED = RES / "promoted.tsv"
if not CAPTSV.exists():
    CAPTSV.write_text("tag\tcell\tmethod\tstage\tmetrics\n")

CONFIGS_DIR = str((Path(__file__).resolve().parents[1] / "configs"))
COMMON = "device=cuda disjoint_extract_refine_data=false extraction_mcq_mode=mc2"
# Keep in sync with scripts/emit_grid_jobs.py CELL_ARGS and scripts/sweep_tqa.sh cell_args().
CELL_ARGS = {
    "ll_qa": "task=truthfulqa eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
    "ll_ch": "task=truthfulqa prompt_template=chat extraction_template=chat eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
    "qw_qa": "task=truthfulqa_qwen eval_batch_size=32 gen_batch_size=8 judge_batch_size=16",
    "qw_ch": "task=truthfulqa_qwen prompt_template=chat extraction_template=chat eval_batch_size=32 gen_batch_size=8 judge_batch_size=16",
    "base_qa": "task=truthfulqa model_name=huggyllama/llama-7b ++model_dtype=float16 eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
}
GEN_BATCH_SIZE = 96      # inspect generative batch (coalescing batcher): 4.7->20.4 prompts/s at bs 8->96.
LOGLIK_BATCH_SIZE = 64   # lm-eval loglik batch. Tuned 2026-07-17 for the 0-shot (short-context) MMLU/ARC:
# throughput peaks at bs=64 (~27 req/s, 17GB), plateaus to bs=256, regresses above. 0-shot is ~6x faster
# per request than 5-shot purely from shorter contexts (the win is 0-shot, not the batch).


def already_done(tag: str) -> bool:
    return any(line.startswith(tag + "\t") for line in CAPTSV.read_text().splitlines())


def fmt_metrics(d: dict) -> str:
    # match sweep_tqa.sh's harvest format: space-separated "KEY: value" with UPPERCASE keys.
    return " ".join(f"{k.upper()}: {v:.4f}" for k, v in d.items() if isinstance(v, (int, float)))


def write_row(tag, cell, method, stage, metrics: dict):
    with open(CAPTSV, "a") as f:
        f.write(f"{tag}\t{cell}\t{method}\t{stage}\t{fmt_metrics(metrics)}\n")


def variants_for(cell, ptag):
    """(tag, stage, kind, kwargs) per capability variant.

    Loglik = FIXED (leaderboard) template ONLY, 0-shot. Chat-template loglik-MC was dropped: lm-eval
    wraps the completion-style primer (…"Answer:") in a chat turn and scores a bare letter as the
    assistant reply, which is not how a chat model answers — it went ~random at 0-shot. The CHAT-template
    capability is instead measured GENERATIVELY (the model actually produces "ANSWER: X" in its own
    voice), so generation is run under BOTH fixed and chat templates.
    """
    v = [
        (f"cap_fxmm_{cell}_{ptag}", "loglik-fx-mmlu", "lmeval",
         dict(tasks=["mmlu"], limit=100, num_fewshot=0, apply_chat_template=False)),
        (f"cap_fxaw_{cell}_{ptag}", "loglik-fx-arcwiki", "lmeval",
         dict(tasks=["arc_challenge", "wikitext"], num_fewshot=0, apply_chat_template=False)),
        (f"cap_genfx_{cell}_{ptag}", "generative-fx", "gen",
         dict(tasks=["mmlu", "arc_challenge"], limit=1000, max_tokens=64, apply_template=False)),
    ]
    if cell != "base_qa":  # base model has no chat template
        v.append((f"cap_genct_{cell}_{ptag}", "generative-ct", "gen",
                  dict(tasks=["mmlu", "arc_challenge"], limit=1000, max_tokens=64, apply_template=True)))
    return v


def cap_points(cell):
    """unsteered baseline + this cell's promoted frontier points -> (ptag, method, args)."""
    pts = [("uns", "unsteered", "method=unsteered")]
    if PROMOTED.exists():
        with open(PROMOTED) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["cell"] == cell:
                    pts.append((r["tag"], r["method"], r["args"]))
    return pts


def build_model(cell, args):
    overrides = f"{COMMON} {CELL_ARGS[cell]} eval_subset_size=2 generative_eval=false {args}".split()
    cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)
    exp = build_experiment(cfg, TruthfulQATask())
    tok = load_tokenizer(exp.config)
    ext_ds, train_ds, _ = exp.task.build_datasets(tok, exp.config)
    model = exp._load_model()
    model, _artifacts, _ci = exp._run_pipeline(model, tok, ext_ds, train_ds, RES / "build")
    model.eval()
    return model, tok


with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
    for cell in CELLS:
        for ptag, method, args in cap_points(cell):
            todo = [v for v in variants_for(cell, ptag) if not already_done(v[0])]
            if not todo:
                print(f"skip {cell}/{ptag} (all variants done)", flush=True)
                continue
            print(f"[caps] build {cell}/{ptag} -> {len(todo)} variants", flush=True)
            model = tok = None
            try:
                model, tok = build_model(cell, args)
            except Exception as e:  # bad config build: log every variant as ERR, keep going
                print(f"ERR build {cell}/{ptag}: {type(e).__name__}: {e}", flush=True)
                continue
            for tag, stage, kind, kw in todo:
                print(f"  CAP {tag}", flush=True)
                try:
                    if kind == "lmeval":
                        m = run_requested_lmeval_tasks(
                            model, tok, kw["tasks"], limit=kw.get("limit"), steer="answer_gen",
                            num_fewshot=kw.get("num_fewshot"),
                            apply_chat_template=kw["apply_chat_template"],
                            fewshot_as_multiturn=kw.get("fewshot_as_multiturn", False),
                            batch_size=LOGLIK_BATCH_SIZE,
                        )
                    else:  # generative (inspect), batched, under the variant's template (fixed/chat)
                        m = run_requested_inspect_evals(
                            model, tok, kw["tasks"], limit=kw["limit"], steer="answer_gen",
                            max_tokens=kw["max_tokens"], gen_batch_size=GEN_BATCH_SIZE,
                            apply_template=kw["apply_template"],
                        )
                    write_row(tag, cell, method, stage, m)
                except Exception as e:
                    print(f"ERR {tag}: {type(e).__name__}: {e}", flush=True)
                # Free THIS variant's cached allocations before the next one. The loglik peak (~32GB on
                # long-context MMLU at bs=64) otherwise stays reserved by the allocator, so the batched
                # gen (bs=96) OOMs even though each fits in isolation.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            free_model_memory()

print(f"[caps] done -> {CAPTSV}", flush=True)

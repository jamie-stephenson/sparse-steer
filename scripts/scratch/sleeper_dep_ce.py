"""TASK 3: teacher-forced clean CE for Table 5.1, on DEPLOYED prompts.

The extra column set: cross-entropy on the completion tokens only (prompt masked out of the
loss), teacher-forced against the CLEAN answer, averaged over the held-out test set, for
Unsteered / Baseline / Sparse across every sleeper.

Row semantics differ deliberately:

    Unsteered   CLEAN prompt, no trigger, no steering  -> the target the others chase
    Baseline    DEPLOYED prompt, steering on           -> how close the fixed ablation gets
    Sparse      DEPLOYED prompt, steering on           -> how close the learned gates get

The completion tokens are identical in all three (the same clean answer), so the numbers are
directly comparable even though the prompts differ in length.

Why a new metric: kl_ce_clean already teacher-forces, masks the prompt out of the loss, scores
completion positions only and averages over the set — but sleeper/eval.py feeds it CLEAN
prompts for every row, so today's clean/ce_steered measures the steering TAX on a normal turn,
not RECOVERY when the trigger is present. kl_ce_clean takes `pairs` as an argument, so feeding
it deployed prompts reuses all of that machinery unchanged and needs no sparse_steer/ edit.

Deployed prompts come from eval_ds's deployed_text — the held-out test split, disjoint from
extraction and gate training, and built by each family's own deploy_text_of (so insertion is
family-correct: prefix for llama/llama2, random in-prompt for qwen, after ": " for tinystories).
They are NOT re-randomised here; that would make the number irreproducible.

Usage (on the pod, from the repo root):
  uv run python scripts/scratch/sleeper_dep_ce.py --models qw
  uv run python scripts/scratch/sleeper_dep_ce.py --models ts,sp,cad,qw --out /root/dep_ce.jsonl
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "run_sleeper_experiments.py"


def load_runner():
    sys.argv = ["x"]
    spec = importlib.util.spec_from_file_location("rse", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def done_readonly(m, args, prefixes: list[str]) -> dict[str, dict]:
    """tag -> cached metrics, built WITHOUT calling harvest().

    harvest() rewrites results.tsv and gates.tsv, which would race the sweep's own writes if
    this runs while the runner is going. cached_metrics is a pure cache lookup (no model load,
    no writes), so the champion/fixed picks can be resolved read-only.
    """
    tags: dict[str, list[str]] = {}
    for tag, (_, ov) in {**m.ts_sparse_grid(), **m.sparse_grid()}.items():
        tags[tag] = ov
    for tag, _stage, ov in m.ordered_jobs({"s1", "s2"}):
        tags.setdefault(tag, ov)
    done = {}
    for tag, ov in tags.items():
        if not any(tag.startswith(p + "_") or tag == f"{p}_unsteered" for p in prefixes):
            continue
        got = m.cached_metrics(args, ov)
        if got is not None:
            done[tag] = got
    return done


def rows_for(m, prefix: str, done: dict) -> list[tuple[str, str, list[str]]]:
    """(row_label, tag, overrides) for the three reported rows of one sleeper."""
    base_task = {"ts": m.TS_B, "sp": m.SP_B, "cad": m.CAD_B, "qw": m.QW_B}[prefix]
    out = [("Unsteered", m.mtag(f"{prefix}_unsteered"),
            m.mov([base_task, "method=unsteered"]))]

    fixed = m.mtag(f"{prefix}_fixed") if prefix == "ts" else m.pick_fixed(done, prefix)
    if fixed:
        s2 = {t: ov for t, _, ov in m.ordered_jobs({"s1", "s2"})}
        if fixed in s2:
            out.append(("Baseline", fixed, s2[fixed]))

    champ = m.pick_champion(done, f"{prefix}_")
    if champ:
        grids = {**m.ts_sparse_grid(), **m.sparse_grid()}
        out.append(("Sparse", champ, grids[champ][1]))
    return out


def build(m, overrides, args):
    """Model + tokenizer for one row, through the runner's own composition path so the
    steering artifacts cache-hit rather than retrain."""
    from sparse_steer.core.loading import load_tokenizer

    exp = m.compose_experiment(m.common(args) + overrides)
    tok = load_tokenizer(exp.config)
    ext_ds, train_ds, eval_ds = exp.task.build_datasets(tok, exp.config)
    model = exp._load_model()
    # Must exist: with track_gates on (the sparse method default, which TinyStories inherits
    # while the big models override it off) _run_pipeline writes gate_heatmap.png here.
    out_dir = Path(args.results_dir) / "dep_ce_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _art, _ci = exp._run_pipeline(model, tok, ext_ds, train_ds, out_dir)
    model.eval()
    return model, tok, eval_ds, exp.config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qw", help="comma list of ts,sp,cad,qw")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", default="sweeps/sleeper")
    ap.add_argument("--out", default="/root/dep_ce.jsonl")
    args = ap.parse_args()

    m = load_runner()
    from hydra import initialize_config_dir

    from sparse_steer.core.kl_provider import kl_ce_clean
    from sparse_steer.tasks.sleeper.data import get_data_module
    from sparse_steer.utils.memory import free_model_memory

    out_path = Path(args.out)
    with initialize_config_dir(config_dir=m.CONFIGS_DIR, version_base=None):
        prefixes = args.models.split(",")
        done = done_readonly(m, args, prefixes)
        print(f"resolved {len(done)} cached tags for {prefixes}", flush=True)
        for prefix in prefixes:
            for label, tag, overrides in rows_for(m, prefix, done):
                print(f"[{time.strftime('%H:%M')}] {prefix} {label} ({tag})", flush=True)
                t0 = time.time()
                model = None
                try:
                    model, tok, eval_ds, cfg = build(m, overrides, args)
                    data = get_data_module(cfg)
                    ctoks = int(cfg.get("completion_tokens", 32))
                    bs = int(cfg.get("jsd_batch_size", 8))
                    # Unsteered scores the CLEAN prompt; the two steered rows score the
                    # DEPLOYED prompt. Completion is the clean answer in every case.
                    key = "clean_text" if label == "Unsteered" else "deployed_text"
                    pairs = [(data.prompt_of(ex[key]), data.completion_of(ex["clean_text"]))
                             for ex in eval_ds]
                    res = kl_ce_clean(model, tok, pairs,
                                      steer=str(cfg.get("steer_token_position", "prompt")),
                                      completion_tokens=ctoks, batch_size=bs)
                    # kl_ce_clean names its outputs clean/*; relabel by what was fed in.
                    tagged = {("dep/" if key == "deployed_text" else "clean/") + k.split("/", 1)[1]: v
                              for k, v in res.items()}
                    row = {"model": prefix, "row": label, "tag": tag,
                           "prompt": "clean" if key == "clean_text" else "deployed",
                           "steer_position": str(cfg.get("steer_token_position", "prompt")),
                           "n_eval": len(pairs), "completion_tokens": ctoks,
                           "seconds": round(time.time() - t0), "metrics": tagged}
                except Exception as e:
                    print(f"ERR {tag}: {type(e).__name__}: {e}", flush=True)
                    row = {"model": prefix, "row": label, "tag": tag,
                           "error": f"{type(e).__name__}: {e}"}
                finally:
                    del model
                    free_model_memory()
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

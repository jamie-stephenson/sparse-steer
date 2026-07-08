"""Exact TL-vs-HF backend comparison for TruthfulQA — the SAME fitted model evaluated twice
in one process on the SAME eval questions (no cache reads/writes for the eval, no subset
tricks). This is the tqa analogue of the tinysleepers validation that matched to 6 decimals.

For the given config overrides it: seeds, builds datasets, runs the (cached) pipeline to a
fitted model, then evaluates run_task_evaluation on the TL engine, switches with
set_backend("hf"), and evaluates again. Prints both metric dicts side by side, their deltas,
and a per-question diff of the saved generations (identical-text count + judge flips).

Usage:  uv run python scripts/compare_backends_tqa.py <hydra overrides...>
e.g.:   ... task=truthfulqa model_name=meta-llama/Llama-2-7b-chat-hf method=sparse \
            eval_subset_size=100 generative_eval=true ...
"""

import csv
import sys
from pathlib import Path

from hydra import compose, initialize
from omegaconf import open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import TASKS  # noqa: E402
from sparse_steer.experiment import build_experiment  # noqa: E402


def _evaluate(exp, model, tokenizer, eval_ds, gen_path: str):
    with open_dict(exp.config):
        exp.config.save_generations_path = gen_path
    return exp.task.run_task_evaluation(model, tokenizer, eval_ds, exp.config)


def main() -> None:
    overrides = sys.argv[1:]
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    task = TASKS[cfg.get("task_name", "truthfulqa")]()
    exp = build_experiment(cfg, task)

    exp._seed_everything(exp.config.seed)
    from sparse_steer.core.loading import load_tokenizer

    tokenizer = load_tokenizer(exp.config)
    extraction_ds, train_ds, eval_ds = exp.task.build_datasets(tokenizer, exp.config)
    model = exp._load_model()
    model, _artifacts, _info = exp._run_pipeline(
        model, tokenizer, extraction_ds, train_ds, Path("output") / "compare_backends"
    )
    model.eval()

    print("=== eval on TL engine ===", flush=True)
    tl_metrics = _evaluate(exp, model, tokenizer, eval_ds, "/tmp/cmp_tl_gen.tsv")
    print("=== switching engine to HF (sdpa) ===", flush=True)
    model.set_backend("hf")
    print("=== eval on HF engine ===", flush=True)
    hf_metrics = _evaluate(exp, model, tokenizer, eval_ds, "/tmp/cmp_hf_gen.tsv")

    print("\nMETRIC                    TL          HF          |delta|")
    for k in sorted(set(tl_metrics) | set(hf_metrics)):
        a, b = tl_metrics.get(k, float("nan")), hf_metrics.get(k, float("nan"))
        print(f"{k:24s}  {a:<10.4f}  {b:<10.4f}  {abs(a - b):.4f}")

    try:
        tl_rows = list(csv.reader(open("/tmp/cmp_tl_gen.tsv"), delimiter="\t"))[1:]
        hf_rows = list(csv.reader(open("/tmp/cmp_hf_gen.tsv"), delimiter="\t"))[1:]
        same_text = sum(1 for a, b in zip(tl_rows, hf_rows) if a[1] == b[1])
        flips = [
            (a[0][:60], a[2:4], b[2:4])
            for a, b in zip(tl_rows, hf_rows)
            if a[2:4] != b[2:4]
        ]
        print(f"\nGENERATIONS: {same_text}/{len(tl_rows)} token-identical")
        print(f"JUDGE FLIPS: {len(flips)}")
        for q, tv, hv in flips:
            print(f"  [{q}] TL T/I={tv} HF T/I={hv}")
    except FileNotFoundError:
        print("\n(no generation TSVs — generative_eval was off)")


if __name__ == "__main__":
    main()

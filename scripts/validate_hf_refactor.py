"""Validate the HF-only refactor against pre-refactor ground truth in <10 min (one pod, 2 GPUs).

Three checks, criteria locked up front:

1. EXTRACTION (tolerance): HF-captured directions vs the TL-era cached STEERING_VECTORS
   artifact for the same cell/fold — per-(component, layer) cosine, PASS min ≥ 0.995.
   (Engines differ numerically like their eval logits do; direction = mean-diff over ~2.3k
   activations, so cosines should sit ≈ 0.999+.)
2. TRAINING (bit): inject the TL-era directions into the refactored cache key, re-run the
   exact pre-refactor 1-epoch gradient smoke via run.py, and require the step-loss trajectory
   to equal /tmp/grad_smoke.log line-for-line. Same engine + same inputs + same seed ⇒ the
   refactor must not have changed a single training bit.
3. EVAL (bit): load the pre-refactor A/B HF-trained artifact (16-epoch, fold 0) and re-run
   the full-set MC eval — MC1/MC2 must equal the A/B run's 0.5281 / 0.7139 exactly.

Phase 2 runs as a subprocess on a second GPU in parallel with phases 1+3 (which share one
model load). The injected directions artifact is removed afterwards so future runs re-extract
under the new lineage.

Usage (pod):  uv run python scripts/validate_hf_refactor.py
Env: GPU_A (phases 1+3, default 0), GPU_B (phase 2, default 1).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

GPU_A = os.environ.get("GPU_A", "0")
GPU_B = os.environ.get("GPU_B", "1")

# The sweep cell this validates against (ll_qa fold 0) — matches the pre-refactor artifacts.
CELL = (
    "device=cuda disjoint_extract_refine_data=false extraction_mcq_mode=mc2 task=truthfulqa "
    "eval_batch_size=64 gen_batch_size=16 judge_batch_size=32 fold=0 method=sparse "
    "train_batch_size=1 +contrastive_weight=1 +ce_weight=0 extract_token_position=completion_final "
    "+contrastive_max_n_neg=3 init_raw_scale=15 l0_lambda=0 gate_config.init_log_alpha=-0.79 "
    "steer_token_position=all compile_models=false"
)
# the pre-refactor 1-epoch gradient smoke (minus the retired +train_backend=hf flag)
SMOKE_OVERRIDES = CELL + (
    " generative_eval=false eval_subset_size=50 track_gates=true num_epochs=1 use_cache=true"
)
GRAD_SMOKE_LOG = Path("/tmp/grad_smoke.log")
AB_MC1, AB_MC2 = "0.5281", "0.7139"  # pre-refactor A/B HF-f0 full-set MC references

MIN_COS = 0.995


def compose_cfg(overrides: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    ctx = initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None)
    ctx.__enter__()  # keep alive for the process lifetime (script, not library)
    cfg = compose(config_name="config", overrides=overrides.split(), return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)
    return cfg


def scan_manifests(**field_filters):
    """Yield (artifact_dir, fields) for every cache manifest matching the field filters.
    A filter value of ("absent",) means the field must NOT be present."""
    for mf in sorted(Path(".cache").glob("**/manifest.json"), key=lambda p: p.stat().st_mtime):
        try:
            fields = json.loads(mf.read_text()).get("config_fields", {})
        except Exception:
            continue
        ok = True
        for k, v in field_filters.items():
            if v == ("absent",):
                ok &= k not in fields
            else:
                ok &= fields.get(k) == v
        if ok:
            yield mf.parent, fields


def find_tl_directions() -> Path:
    """The TL-era STEERING_VECTORS artifact for ll_qa fold 0 (no _engine lineage field)."""
    hits = [
        d for d, _ in scan_manifests(
            _task_name="truthfulqa",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            extract_token_position="completion_final",
            _engine=("absent",),
            fold=("absent",),  # fold 0 is keyed by ABSENCE (default-fold artifacts carry no key)
        )
        if (d / "steering_vectors.pt").is_file()
    ]
    if not hits:
        sys.exit("FAIL: no TL-era steering_vectors artifact found in .cache")
    return hits[-1] / "steering_vectors.pt"  # most recent


def find_ab_artifact() -> Path:
    """The pre-refactor A/B HF-trained (train_backend=hf, 16-epoch) steering artifact."""
    hits = [
        d for d, f in scan_manifests(
            _task_name="truthfulqa",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            train_backend="hf",
            fold=("absent",),  # the A/B trained fold 0 (default fold carries no key)
        )
        if f.get("num_epochs") == 16 and (d / "steering.pt").is_file()
    ]
    if not hits:
        sys.exit("FAIL: no A/B train_backend=hf steering artifact found in .cache")
    return hits[-1] / "steering.pt"


def main() -> None:
    t0 = time.monotonic()
    from sparse_steer.experiment import build_experiment
    from sparse_steer.tasks.truthfulqa.task import TruthfulQATask
    from sparse_steer.core.extract import (
        collect_activations, extract_steering_vectors, load_steering_vectors,
        save_steering_vectors,
    )
    from sparse_steer.core.loading import load_steering_model, load_tokenizer
    from sparse_steer.utils.cache import ArtifactType
    from sparse_steer.utils.compile import set_compile

    tl_dirs_path = find_tl_directions()
    ab_path = find_ab_artifact()
    print(f"[refs] TL directions: {tl_dirs_path}")
    print(f"[refs] A/B artifact : {ab_path}")
    if not GRAD_SMOKE_LOG.is_file():
        sys.exit(f"FAIL: reference trajectory {GRAD_SMOKE_LOG} missing")

    # ── Phase 2 setup: inject TL directions under the refactored cache key, launch run.py ──
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_A
    set_compile(False)
    cfg = compose_cfg(CELL + " generative_eval=false num_epochs=1 use_cache=true track_gates=true eval_subset_size=50")
    task = TruthfulQATask()
    exp = build_experiment(cfg, task)
    dest = exp._prepare_cache_path(ArtifactType.STEERING_VECTORS)
    vectors, meta = load_steering_vectors(tl_dirs_path)
    save_steering_vectors(vectors, dest, metadata=dict(meta, injected_from=str(tl_dirs_path)))
    injected_dir = exp._finalize_cache(ArtifactType.STEERING_VECTORS)
    print(f"[2] injected TL directions -> {injected_dir}")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_B)
    train_log = open("/tmp/val_train.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "run.py"] + SMOKE_OVERRIDES.split(),
        stdout=train_log, stderr=subprocess.STDOUT, env=env, cwd=REPO,
    )
    print(f"[2] train-parity run launched on GPU {GPU_B} (pid {proc.pid})")

    # ── Phase 1: fresh HF extraction vs TL directions (GPU_A, this process) ──
    tokenizer = load_tokenizer(cfg)
    extraction_ds, train_ds, eval_ds = task.build_datasets(tokenizer, cfg)
    model = load_steering_model(cfg)
    with_acts, comps = collect_activations(
        extraction_ds, model, tokenizer,
        targets=list(cfg.targets), batch_size=cfg.extract_batch_size,
        token_position=cfg.extract_token_position,
    )
    hf_dirs = extract_steering_vectors(with_acts, comps)
    worst, report = 1.0, []
    for comp, tl_v in vectors.items():
        hf_v = hf_dirs[comp].float()
        tl_v = tl_v.float()
        cos = F.cosine_similarity(hf_v.flatten(1), tl_v.flatten(1), dim=1)  # per layer
        worst = min(worst, float(cos.min()))
        report.append(f"    {comp}: min={cos.min():.6f} mean={cos.mean():.6f}")
    print("[1] extraction cosine (HF-captured vs TL-era directions):")
    print("\n".join(report))
    p1 = worst >= MIN_COS
    print(f"[1] {'PASS' if p1 else 'FAIL'}: min cosine {worst:.6f} (bar {MIN_COS})")

    # ── Phase 3: bit-parity eval of the A/B artifact (same loaded model) ──
    from sparse_steer.tasks.truthfulqa.eval import evaluate

    model.load_steering(ab_path)
    model.eval()
    with torch.no_grad():
        mc = evaluate(
            model, tokenizer, eval_ds, batch_size=int(cfg.eval_batch_size),
            steer_token_position=str(cfg.steer_token_position), template=str(cfg.prompt_template),
        )
    got1, got2 = f"{mc['mc1']:.4f}", f"{mc['mc2']:.4f}"
    p3 = (got1, got2) == (AB_MC1, AB_MC2)
    print(f"[3] {'PASS' if p3 else 'FAIL'}: MC1 {got1} (ref {AB_MC1}), MC2 {got2} (ref {AB_MC2})")

    # ── Phase 2: collect + compare trajectories ──
    rc = proc.wait()
    train_log.close()
    def steps(path):
        import re
        txt = Path(path).read_text(errors="replace").replace("\r", "\n")
        return re.findall(r"step \d+: loss=[0-9.]+", txt)
    ref, got = steps(GRAD_SMOKE_LOG), steps("/tmp/val_train.log")
    p2 = rc == 0 and len(got) > 0 and ref == got
    if not p2 and len(ref) != len(got):
        print(f"[2] step-count mismatch: ref {len(ref)} vs got {len(got)} (rc={rc})")
    elif not p2:
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), -1)
        print(f"[2] first divergence at index {first}: ref={ref[first]!r} got={got[first]!r}")
    print(f"[2] {'PASS' if p2 else 'FAIL'}: {len(got)}/{len(ref)} training steps bit-identical (rc={rc})")

    # cleanup: drop the injected artifact so future runs re-extract under the new lineage
    shutil.rmtree(injected_dir if injected_dir.is_dir() else injected_dir.parent, ignore_errors=True)
    print(f"[cleanup] removed injected directions {injected_dir}")

    ok = p1 and p2 and p3
    print(f"\n=== VALIDATION {'PASS' if ok else 'FAIL'} in {time.monotonic() - t0:.0f}s ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

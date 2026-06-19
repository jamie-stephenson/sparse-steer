"""The SPARSE_STEERING (and gate-training STEERED_EVAL) cache key must capture the whole
training recipe — otherwise a frontier sweep over l0_lambda silently reuses one result."""

from omegaconf import OmegaConf

from sparse_steer.tasks.jailbreak.task import RefusalTask
from sparse_steer.utils.cache import ArtifactType, compute_config_hash

TASK = RefusalTask()


def _cfg(**overrides):
    base = dict(
        dtype="float16", lora_adapter=None,
        harmful_dataset="advbench", harmless_dataset="alpaca",
        data_origin="arditi_exact", data_mix=None,
        n_extraction=256, extraction_subset="category",
        gen_decoding="greedy", gen_max_new_tokens=512, gen_temperature=1.0, gen_seed=0,
        refusal_detector="logit", refusal_tokens=["I", "As"], refusal_logit_threshold=0.0,
        intervention="ablate", direction_source="self",
        targets=["resid_pre", "resid_mid", "resid_post"], extract_token_position="last",
        steering_layer_ids=None, normalize_steering_vectors=False,
        refinement_method="gate_training",
        n_train=64, affirmative_prefix="Sure, here is", completion_tokens=64, seed=42,
        num_epochs=2, learning_rate=1.5e-2, lr_scheduler_type="cosine", lr_warmup_steps=5,
        weight_decay=0.01, train_batch_size=8,
        l0_lambda=0.04, l0_scheduler_type="warmup", l0_warmup_steps=16,
        normalize_ablation=True, proj_act_norm_examples=128,
        learn_scale=True, shared_scale=False, init_raw_scale=0.5413, freeze_raw_scale=False,
        scale_tuning_epochs=0, scale_tuning_lr=None,
        gate_config={"temperature": 0.33, "stretch_limits": [-0.1, 1.1], "eps": 1e-6,
                     "eval_threshold": 1e-2, "init_log_alpha": -0.79},
        # eval block (STEERED_EVAL); harmful/harmless_eval set so judge_eval_dataset isn't needed
        judge="regex", n_eval=200, harmful_eval_dataset="advbench", harmless_eval_dataset="alpaca",
        evals=["safety_score"], generative_eval=True, gen_tokens=512, eval_seeds=[0],
        eval_temperature=0, steer_token_position="all", eval_refusal_detector="arditi",
        llama_guard_model="meta-llama/Meta-Llama-Guard-2-8B",
        selection_grid_component="resid_pre", selection_positions=5,
        selection_kl_threshold=0.1, selection_induce_threshold=0.0, selection_prune_layer_frac=0.2,
    )
    base.update(overrides)
    return OmegaConf.create(base)


def _key(artifact, **overrides):
    """The task's *extra* cache fields only."""
    return TASK.extra_cache_fields(artifact, _cfg(**overrides))


def _hash(artifact, **overrides):
    """The full cache key (base _CONFIG_FIELDS + the task's extra). Training-recipe knobs may
    live in either, so the real 'is it in the key' contract is tested against this."""
    cfg = _cfg(**overrides)
    return compute_config_hash(cfg, artifact, TASK.task_name, TASK.extra_cache_fields(artifact, cfg))[0]


def test_l0_lambda_changes_sparse_steering_key():
    # the headline bug: the frontier sweep knob must be in the key
    assert _hash(ArtifactType.SPARSE_STEERING, l0_lambda=0.04) != _hash(
        ArtifactType.SPARSE_STEERING, l0_lambda=0.08
    )


def test_training_knobs_change_sparse_steering_key():
    base = _hash(ArtifactType.SPARSE_STEERING)
    for knob, value in [
        ("num_epochs", 20), ("normalize_ablation", False), ("learning_rate", 3e-2),
        ("init_raw_scale", 5.0), ("direction_source", ["resid_pre", 17]), ("seed", 7),
    ]:
        assert _hash(ArtifactType.SPARSE_STEERING, **{knob: value}) != base, f"{knob} not in key"


def test_l0_lambda_changes_gate_training_eval_key():
    # the eval that consumes the trained gates must also re-key on the training recipe
    assert _hash(ArtifactType.STEERED_EVAL, l0_lambda=0.04) != _hash(
        ArtifactType.STEERED_EVAL, l0_lambda=0.08
    )


def test_grid_search_eval_key_ignores_gate_training_knobs():
    # a no-training selection run's *extra* must NOT carry gate-training knobs (e.g.
    # normalize_ablation): the gate block is gated out, so they never enter its identity.
    a = _key(ArtifactType.STEERED_EVAL, refinement_method="none", direction_source="grid_search", normalize_ablation=True)
    b = _key(ArtifactType.STEERED_EVAL, refinement_method="none", direction_source="grid_search", normalize_ablation=False)
    assert a == b

"""Artifact caching with staleness detection for the sparse-steering pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

# ── Cache root ────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = _PROJECT_ROOT / ".cache" / "sparse_steer"

# ── Artifact types ────────────────────────────────────────────────────


class ArtifactType(Enum):
    STEERING_VECTORS = "steering_vectors"
    SPARSE_STEERING = "sparse_steering"
    PEFT_ADAPTER = "peft_adapter"
    BASELINE_EVAL = "baseline_eval"
    STEERED_EVAL = "steered_eval"


# Config fields that form the cache key for each artifact type.
# Task-specific extras are appended by the experiment subclass.
_CONFIG_FIELDS: dict[ArtifactType, list[str]] = {
    ArtifactType.STEERING_VECTORS: [
        "model_name",
        "seed",
        "extraction_fraction",
        "extract_batch_size",
        "token_position",
        "targets",
    ],
    ArtifactType.SPARSE_STEERING: [
        "method",
        # inherits steering-vector fields
        "model_name",
        "seed",
        "extraction_fraction",
        "extract_batch_size",
        "token_position",
        "targets",
        "steering_layer_ids",
        # training-specific
        "normalize_steering_vectors",
        "l0_scheduler_type",
        "l0_warmup_steps",
        "l0_lambda",
        "learning_rate",
        "lr_scheduler_type",
        "lr_warmup_steps",
        "num_epochs",
        "train_batch_size",
        "weight_decay",
        "freeze_log_scale",
        "gate_config",
    ],
    ArtifactType.BASELINE_EVAL: [
        "model_name",
        "seed",
        "eval_batch_size",
    ],
    ArtifactType.PEFT_ADAPTER: [
        "method",
        "model_name",
        "seed",
        "extraction_fraction",
        # training
        "learning_rate",
        "lr_scheduler_type",
        "lr_warmup_steps",
        "num_epochs",
        "train_batch_size",
        "weight_decay",
        # lora-specific
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "lora_dropout",
    ],
    ArtifactType.STEERED_EVAL: [
        "method",
        "model_name",
        "seed",
        "extraction_fraction",
        "eval_batch_size",
        # steering fields (None for non-steering methods)
        "steering_strength",
        "extract_batch_size",
        "token_position",
        "targets",
        "steering_layer_ids",
        "normalize_steering_vectors",
        # sparse training fields (None for non-sparse methods)
        "l0_scheduler_type",
        "l0_warmup_steps",
        "l0_lambda",
        "learning_rate",
        "lr_scheduler_type",
        "lr_warmup_steps",
        "num_epochs",
        "train_batch_size",
        "weight_decay",
        "freeze_log_scale",
        "gate_config",
        # lora fields (None for non-lora methods)
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "lora_dropout",
    ],
}

# Source files whose content is hashed for staleness detection.
# Paths are relative to the project root.
_SOURCE_FILES: dict[ArtifactType, list[str]] = {
    ArtifactType.STEERING_VECTORS: [
        "sparse_steer/extract.py",
        "sparse_steer/models/base.py",
    ],
    ArtifactType.SPARSE_STEERING: [
        "sparse_steer/extract.py",
        "sparse_steer/models/base.py",
        "sparse_steer/train.py",
        "sparse_steer/hardconcrete.py",
        "sparse_steer/models/sparse.py",
    ],
    ArtifactType.PEFT_ADAPTER: [
        "sparse_steer/train.py",
    ],
    ArtifactType.BASELINE_EVAL: [
        "sparse_steer/utils/eval.py",
    ],
    ArtifactType.STEERED_EVAL: [
        "sparse_steer/extract.py",
        "sparse_steer/models/base.py",
        "sparse_steer/train.py",
        "sparse_steer/hardconcrete.py",
        "sparse_steer/models/sparse.py",
        "sparse_steer/utils/eval.py",
    ],
}

# Artifact filename stored in cache
_ARTIFACT_FILENAMES: dict[ArtifactType, str] = {
    ArtifactType.STEERING_VECTORS: "steering_vectors.pt",
    ArtifactType.SPARSE_STEERING: "sparse_steering.pt",
    ArtifactType.PEFT_ADAPTER: "adapter_config.json",
    ArtifactType.BASELINE_EVAL: "results.json",
    ArtifactType.STEERED_EVAL: "results.json",
}


# ── Manifest ──────────────────────────────────────────────────────────


@dataclass
class CacheManifest:
    artifact_type: str
    config_hash: str
    source_hash: str
    git_commit: str
    created_at: str
    config_fields: dict[str, Any]
    source_files: list[str]
    per_file_hashes: dict[str, str]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> CacheManifest:
        return cls(**json.loads(path.read_text()))


@dataclass
class CacheHit:
    artifact_path: Path
    manifest: CacheManifest
    warnings: list[str] = field(default_factory=list)


# ── Hashing helpers ───────────────────────────────────────────────────


def _normalize_for_hash(value: Any) -> Any:
    """Normalize values for deterministic JSON serialisation."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(value, (DictConfig, dict)):
        return {k: _normalize_for_hash(v) for k, v in sorted(value.items())}
    if isinstance(value, (ListConfig, list)):
        return [_normalize_for_hash(v) for v in value]
    return value


def compute_config_hash(
    config: "DictConfig",
    artifact_type: ArtifactType,
    task_name: str,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (hex_sha256, fields_dict) for the given artifact type."""
    field_names = _CONFIG_FIELDS[artifact_type]
    fields_dict: dict[str, Any] = {"_task_name": task_name}
    for name in field_names:
        value = getattr(config, name, None)
        # sort targets list (order-insensitive)
        if name == "targets" and isinstance(value, list):
            value = sorted(value)
        fields_dict[name] = value
    if extra_fields:
        fields_dict.update(extra_fields)

    normalized = _normalize_for_hash(fields_dict)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest(), normalized


def compute_source_hash(
    artifact_type: ArtifactType,
    extra_sources: list[str] | None = None,
) -> tuple[str, dict[str, str], list[str]]:
    """Return (combined_hex_sha256, per_file_hashes, file_list)."""
    files = list(_SOURCE_FILES[artifact_type])
    if extra_sources:
        files.extend(extra_sources)
    files = sorted(set(files))

    hasher = hashlib.sha256()
    per_file: dict[str, str] = {}
    for rel in files:
        abs_path = _PROJECT_ROOT / rel
        if abs_path.is_file():
            content = abs_path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
        else:
            file_hash = "MISSING"
        per_file[rel] = file_hash
        hasher.update(f"{rel}:{file_hash}\n".encode())

    return hasher.hexdigest(), per_file, files


# ── Git helpers ───────────────────────────────────────────────────────


def current_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _commits_since(old_commit: str) -> int | None:
    if old_commit == "unknown":
        return None
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-list", "--count", f"{old_commit}..HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return int(out)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


# ── Staleness detection ───────────────────────────────────────────────

_COMMIT_DISTANCE_THRESHOLD = 10


def check_staleness(
    manifest: CacheManifest,
    artifact_type: ArtifactType,
    extra_sources: list[str] | None = None,
) -> list[str]:
    """Return a list of warning strings (empty = fresh)."""
    warnings: list[str] = []

    # 1. Source code drift
    current_hash, current_per_file, _ = compute_source_hash(
        artifact_type, extra_sources
    )
    if current_hash != manifest.source_hash:
        changed = [
            f
            for f in manifest.per_file_hashes
            if manifest.per_file_hashes.get(f) != current_per_file.get(f)
        ]
        # also check for new files not in manifest
        for f in current_per_file:
            if f not in manifest.per_file_hashes:
                changed.append(f)
        changed = sorted(set(changed))
        warnings.append(
            "Source code has changed since this artifact was cached.\n"
            f"  Cached at commit {manifest.git_commit}, current commit {current_git_commit()}.\n"
            "  Changed files:\n" + "".join(f"    - {f}\n" for f in changed)
        )

    # 2. Commit distance
    distance = _commits_since(manifest.git_commit)
    if distance is not None and distance > _COMMIT_DISTANCE_THRESHOLD:
        warnings.append(
            f"Cached artifact is {distance} commits old (from {manifest.created_at})."
        )

    return warnings


def print_cache_warnings(
    artifact_label: str,
    warnings: list[str],
) -> None:
    """Print a visible warning banner to stderr."""
    if not warnings:
        return

    lines = [f"CACHE WARNING: {artifact_label}"]
    for w in warnings:
        for line in w.rstrip().split("\n"):
            lines.append(line)
    lines.append("")
    lines.append("Set use_cache: false in config to force recomputation.")

    width = max(len(line) for line in lines) + 2
    border = "─" * (width + 2)

    print(f"\n┌{border}┐", file=sys.stderr)
    for line in lines:
        print(f"│ {line.ljust(width)} │", file=sys.stderr)
    print(f"└{border}┘\n", file=sys.stderr)


# ── Lookup / Store ────────────────────────────────────────────────────


def _cache_dir(artifact_type: ArtifactType, task_name: str, config_hash: str) -> Path:
    return CACHE_ROOT / artifact_type.value / task_name / config_hash[:8]


def lookup(
    artifact_type: ArtifactType,
    config: "DictConfig",
    task_name: str,
    *,
    extra_fields: dict[str, Any] | None = None,
    extra_sources: list[str] | None = None,
) -> CacheHit | None:
    """Check for a cached artifact. Returns CacheHit or None."""
    config_hash, _ = compute_config_hash(config, artifact_type, task_name, extra_fields)
    cache_path = _cache_dir(artifact_type, task_name, config_hash)
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = CacheManifest.load(manifest_path)
    # verify full hash (prefix collision guard)
    if manifest.config_hash != config_hash:
        return None

    artifact_path = cache_path / _ARTIFACT_FILENAMES[artifact_type]
    if not artifact_path.is_file():
        return None

    warnings = check_staleness(manifest, artifact_type, extra_sources)
    return CacheHit(
        artifact_path=artifact_path,
        manifest=manifest,
        warnings=warnings,
    )


def prepare_cache_path(
    artifact_type: ArtifactType,
    config: "DictConfig",
    task_name: str,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    """Return the path where an artifact should be saved. Creates parent dirs."""
    config_hash, _ = compute_config_hash(config, artifact_type, task_name, extra_fields)
    cache_path = _cache_dir(artifact_type, task_name, config_hash)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / _ARTIFACT_FILENAMES[artifact_type]


def finalize(
    artifact_type: ArtifactType,
    config: "DictConfig",
    task_name: str,
    *,
    extra_fields: dict[str, Any] | None = None,
    extra_sources: list[str] | None = None,
) -> Path:
    """Write the cache manifest after the artifact has been saved. Returns artifact path."""
    config_hash, config_fields = compute_config_hash(
        config,
        artifact_type,
        task_name,
        extra_fields,
    )
    source_hash, per_file_hashes, source_files = compute_source_hash(
        artifact_type,
        extra_sources,
    )

    cache_path = _cache_dir(artifact_type, task_name, config_hash)
    dest = cache_path / _ARTIFACT_FILENAMES[artifact_type]

    manifest = CacheManifest(
        artifact_type=artifact_type.value,
        config_hash=config_hash,
        source_hash=source_hash,
        git_commit=current_git_commit(),
        created_at=datetime.now(timezone.utc).isoformat(),
        config_fields=config_fields,
        source_files=source_files,
        per_file_hashes=per_file_hashes,
    )
    manifest.save(cache_path / "manifest.json")
    return dest


def store_json(
    artifact_type: ArtifactType,
    config: "DictConfig",
    task_name: str,
    data: dict[str, Any],
    *,
    extra_fields: dict[str, Any] | None = None,
    extra_sources: list[str] | None = None,
) -> Path:
    """Store a JSON-serialisable dict (eval results) in the cache."""
    dest = prepare_cache_path(
        artifact_type, config, task_name, extra_fields=extra_fields
    )
    dest.write_text(json.dumps(data, indent=2))
    return finalize(
        artifact_type,
        config,
        task_name,
        extra_fields=extra_fields,
        extra_sources=extra_sources,
    )


def load_cached_json(path: Path) -> dict[str, Any]:
    """Load a cached JSON artifact."""
    return json.loads(path.read_text())


__all__ = [
    "ArtifactType",
    "CacheHit",
    "CacheManifest",
    "check_staleness",
    "compute_config_hash",
    "compute_source_hash",
    "current_git_commit",
    "finalize",
    "load_cached_json",
    "lookup",
    "prepare_cache_path",
    "print_cache_warnings",
    "store_json",
]

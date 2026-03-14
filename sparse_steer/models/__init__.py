from .base import BaseSteeringLM
from .sparse import SparseSteeringLM
from .dense import DenseSteeringLM
from .llama import LlamaSparseSteeringLM, LlamaDenseSteeringLM
from .qwen2 import Qwen2SparseSteeringLM, Qwen2DenseSteeringLM

MODEL_REGISTRY: dict[str, type[SparseSteeringLM]] = {
    "llama": LlamaSparseSteeringLM,
    "qwen2": Qwen2SparseSteeringLM,
}

DENSE_MODEL_REGISTRY: dict[str, type[DenseSteeringLM]] = {
    "llama": LlamaDenseSteeringLM,
    "qwen2": Qwen2DenseSteeringLM,
}

__all__ = [
    "BaseSteeringLM",
    "DENSE_MODEL_REGISTRY",
    "DenseSteeringLM",
    "LlamaDenseSteeringLM",
    "LlamaSparseSteeringLM",
    "MODEL_REGISTRY",
    "Qwen2DenseSteeringLM",
    "Qwen2SparseSteeringLM",
    "SparseSteeringLM",
]

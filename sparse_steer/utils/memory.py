"""Accelerator memory hygiene (dependency leaf: torch only)."""

import gc

import torch


def free_model_memory() -> None:
    """Release cached accelerator memory after ``del``-ing a model.

    Canonical replacement for the hand-rolled ``gc.collect()`` + ``empty_cache`` idiom
    (previously three copies with inconsistent cuda/mps ordering and one missing the mps
    branch). The caller ``del``s its reference first; this collects and empties whichever
    accelerator cache exists. cuda/mps are mutually exclusive so branch order is moot.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

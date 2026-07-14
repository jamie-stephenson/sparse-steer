"""Single switch for torch.compile across the model + judges.

Compilation is ON by default (config ``compile_models: true``) and applied wherever a model is loaded
via :func:`maybe_compile`. The experiment sets the flag once from config at startup, so no env vars are
involved. Set ``compile_models=false`` for a bit-faithful / debugging run (analogous to
``iti_probe_device=cpu``) or if a future runtime makes compilation unsafe — the compile smoke test
validates that compilation leaves steered outputs unchanged.
"""
_ENABLED = True


def set_compile(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def maybe_compile(model):
    """torch.compile(model) when enabled and the runtime supports it; otherwise return it unchanged."""
    if not _ENABLED:
        return model
    try:
        import torch

        return torch.compile(model, mode="reduce-overhead")
    except Exception as exc:  # old torch / unsupported model -> run uncompiled
        print(f"  torch.compile unavailable ({exc}); running uncompiled.")
        return model


__all__ = ["set_compile", "maybe_compile"]

"""Native-HF eval backend for :class:`SteeringModel` — the fast (sdpa) counterpart of the
TransformerLens engine.

TL's instrumented eager-attention forward is ~2-5x slower than native HF with sdpa/flash.
This module lets the SAME ``SteeringModel`` (same :class:`SteeringHook` modules — vectors,
hard-concrete gates, scales, ``proj_act_norm``, intervention) run on a plain
``AutoModelForCausalLM``: :class:`HFHookAdapter` registers HF forward-hooks that call the
*identical* ``SteeringHook.steer`` / ``SteeringHook.ablate`` functions on the equivalent
activation tensors, so the math is shared, not reimplemented.

TL hook site → HF module boundary (per supported family):

====================  ============================================================
TL site               HF edit point
====================  ============================================================
``resid_pre``         forward-pre-hook on ``layers[i]`` (edit the block input)
``resid_mid``         the block exposes no module whose output IS resid_mid, so it is
                      reconstructed: a pre-hook on ``layers[i]`` captures the (possibly
                      resid_pre-edited) block input ``residual``; a forward-hook on the
                      attention module then forms ``h_mid = residual + attn_out``, applies
                      the edit, and returns ``attn_out' = edited(h_mid) − residual`` — the
                      block's own ``residual + attn_out'`` reproduces the edited resid_mid
                      exactly, for BOTH steer and ablate (the projection sees the full
                      summed stream, matching TL's ``hook_resid_mid``).
``resid_post``        forward-hook on ``layers[i]`` (edit the block output)
``attention`` (z)     forward-pre-hook on the attention out-projection (``o_proj`` /
                      ``out_proj``): its input is TL's z flattened ``(B, T, H*dh)``
                      (position-major, head-contiguous in both HF and TL)
``attn_out``          forward-hook on the attention module (edit its output hidden states)
``mlp`` (hook_post)   forward-pre-hook on the MLP down-projection (``down_proj`` /
                      ``c_proj``): its input is TL's post-activation ``(B, T, d_mlp)``
====================  ============================================================

Hook ordering within a block matches TL (z → attn_out → resid_mid → mlp post → resid_post):
pre-hooks/forward-hooks chain in registration order and each sees the previous edit.

Eval-mode gates that evaluate to 0 (below the hard-concrete eval threshold) make a site a
strict no-op, so those sites are skipped ENTIRELY at wiring time — the efficiency win for
sparse solutions. ``SteeringModel.load_steering`` / ``set_all_vectors`` call
:meth:`HFHookAdapter.rewire` so the skip set stays in sync if the steering state changes.

Position masks under HF KV-cache generation: a ``steer_positions`` mask whose width equals
the current activation's sequence length applies directly (the TL-cache convention used by
``core.generate``); a mask of a DIFFERENT width is interpreted over *absolute* positions and
sliced per chunk using the current cache offset (read from ``cache_position`` /
``past_key_values`` by a top-level pre-hook), with out-of-range positions unsteered — so a
prompt-width mask steers only the prompt under ``model.generate`` decode steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:  # avoid a circular import (steering imports this lazily)
    from .steering import SteeringHook, SteeringModel


# ── Loading ───────────────────────────────────────────────────────────


def load_hf_model(
    model_name: str,
    *,
    lora_adapter: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Load the plain HF causal LM for ``model_name`` (same checkpoint cache as the TL load).

    Mirrors ``load_hooked_transformer``'s weight recipe: optional LoRA merge into the base,
    ``trust_remote_code`` for custom checkpoints. ``architecture_name`` is a TL-only concept
    (TL needs a known config name); the HF side always loads ``model_name`` directly.
    Prefers sdpa attention (the speed win) with a fallback for models that reject it.
    """
    kwargs: dict = {"torch_dtype": dtype, "trust_remote_code": True}
    if device not in (None, "cpu") and lora_adapter is None:
        # Load straight onto the accelerator: avoids a transient full CPU copy (the same
        # discipline as the TL split_source load path).
        kwargs["device_map"] = {"": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="sdpa", **kwargs
        )
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if lora_adapter is not None:
        from peft import PeftModel

        print(f"Merging LoRA adapter '{lora_adapter}' into base '{model_name}' (hf backend)...")
        model = PeftModel.from_pretrained(model, lora_adapter).merge_and_unload()
    if device is not None:
        model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


# ── Architecture family mapping ───────────────────────────────────────


class _Family:
    """Module paths for one architecture family (how to find each edit point)."""

    def __init__(
        self,
        layers: Callable[[nn.Module], nn.ModuleList],
        attn: Callable[[nn.Module], nn.Module],
        z_proj: Callable[[nn.Module], nn.Module],
        mlp_in: Callable[[nn.Module], nn.Module],
    ) -> None:
        self.layers = layers
        self.attn = attn        # module whose output hidden states are attn_out
        self.z_proj = z_proj    # out-projection whose INPUT is z flattened
        self.mlp_in = mlp_in    # down-projection whose INPUT is TL's mlp hook_post


_FAMILIES: dict[str, _Family] = {
    # Llama-family (Llama-2/3, Mistral, Qwen2/2.5/3): model.model.layers[i]
    "llama": _Family(
        layers=lambda m: m.model.layers,
        attn=lambda b: b.self_attn,
        z_proj=lambda b: b.self_attn.o_proj,
        mlp_in=lambda b: b.mlp.down_proj,
    ),
    # GPT-Neo (TinyStories family): model.transformer.h[i]
    "gpt_neo": _Family(
        layers=lambda m: m.transformer.h,
        attn=lambda b: b.attn,
        z_proj=lambda b: b.attn.attention.out_proj,
        mlp_in=lambda b: b.mlp.c_proj,
    ),
}
_FAMILY_ALIASES = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "llama",
    "qwen3": "llama",
    "gpt_neo": "gpt_neo",
}


def _resolve_family(model: nn.Module) -> _Family:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    key = _FAMILY_ALIASES.get(model_type)
    if key is None:
        raise RuntimeError(
            f"hf backend does not support architecture {model_type!r} yet; supported: "
            f"{sorted(_FAMILY_ALIASES)}. Use eval_backend=tl for this model."
        )
    return _FAMILIES[key]


# ── Adapter ───────────────────────────────────────────────────────────


def _site_is_active(hook: "SteeringHook") -> bool:
    """True unless the site is a strict no-op in eval mode.

    A zero steering vector contributes nothing under either intervention (steer adds 0;
    ablate projects onto a zero direction), and an eval-mode ``gate*scale`` of exactly 0 at
    every gate zeroes the correction. Skipping such sites is unobservable — and the whole
    point: a sparse solution touches a handful of modules instead of every layer.
    """
    if int(hook.steering_vectors.count_nonzero()) == 0:
        return False
    was_training = hook.training
    hook.eval()
    try:
        weight = hook._gate_weights() * hook._scale_weights()
    finally:
        hook.train(was_training)
    if isinstance(weight, Tensor):
        return bool((weight != 0).any())
    return weight != 0


class HFHookAdapter:
    """Registers HF module hooks that apply an owner ``SteeringModel``'s steering state.

    Reads the SAME :class:`SteeringHook` modules the TL engine uses (single source of
    truth); nothing is copied or converted. ``remove()`` detaches every handle (called
    before the HF engine is freed); ``rewire()`` re-registers after a steering-state change.
    """

    def __init__(self, owner: "SteeringModel") -> None:
        self.owner = owner
        self.model = owner.hf
        self.family = _resolve_family(self.model)
        self._handles: list = []
        self._resid_pre: dict[int, Tensor] = {}
        self._chunk_start = 0
        self.active_sites: list[str] = []
        self._wire()

    # ── lifecycle ────────────────────────────────────────────────────

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._resid_pre.clear()

    def rewire(self) -> None:
        self.remove()
        self._wire()

    # ── the shared edit application ──────────────────────────────────

    def _apply(self, hook: "SteeringHook", act: Tensor) -> Tensor:
        """Run the hook's own steer/ablate on ``act`` (identical math to the TL engine),
        translating the position mask for KV-cache chunks when widths differ."""
        fn = hook.ablate if self.owner.intervention == "ablate" else hook.steer
        mask = hook.pos_mask
        if mask is None or mask.shape[1] == act.shape[1]:
            return fn(act)
        # Width mismatch → the mask is over absolute positions; slice out this chunk
        # (positions beyond the mask are unsteered — e.g. decode steps past a prompt mask).
        start = self._chunk_start
        chunk = mask.new_zeros(mask.shape[0], act.shape[1])
        lo, hi = max(start, 0), min(start + act.shape[1], mask.shape[1])
        if hi > lo:
            chunk[:, lo - start : hi - start] = mask[:, lo:hi]
        hook.pos_mask = chunk
        try:
            return fn(act)
        finally:
            hook.pos_mask = mask

    # ── wiring ───────────────────────────────────────────────────────

    def _wire(self) -> None:
        owner = self.owner
        blocks = self.family.layers(self.model)
        per_layer: dict[int, dict[str, "SteeringHook"]] = {}
        self.active_sites = []
        for component, layer, hook in owner.iter_hooks():
            if not _site_is_active(hook):
                continue
            per_layer.setdefault(layer, {})[component] = hook
            self.active_sites.append(f"{component}_{layer}")

        # Track the absolute start position of each forwarded chunk (for absolute-position
        # mask slicing under KV-cache generation). cache_position is authoritative when the
        # forward provides it (model.generate does); else fall back to the cache length.
        if per_layer:
            def track_chunk(module, args, kwargs):
                cp = kwargs.get("cache_position")
                if cp is not None and len(cp) > 0:
                    self._chunk_start = int(cp[0])
                    return None
                past = kwargs.get("past_key_values")
                if past is not None:
                    try:
                        self._chunk_start = int(past.get_seq_length())
                        return None
                    except (AttributeError, TypeError):
                        pass
                self._chunk_start = 0
                return None

            self._handles.append(
                self.model.register_forward_pre_hook(track_chunk, with_kwargs=True)
            )

        for layer, comps in sorted(per_layer.items()):
            block = blocks[layer]
            self._wire_block(layer, block, comps)

    def _wire_block(self, layer: int, block: nn.Module, comps: dict) -> None:
        resid_pre = comps.get("resid_pre")
        resid_mid = comps.get("resid_mid")
        resid_post = comps.get("resid_post")
        attn_out = comps.get("attn_out")
        attention = comps.get("attention")
        mlp = comps.get("mlp")

        if resid_pre is not None or resid_mid is not None:
            def block_pre(module, args, kwargs, *, _layer=layer, _pre=resid_pre, _need_mid=resid_mid is not None):
                hs = args[0] if args else kwargs["hidden_states"]
                if _pre is not None:
                    hs = self._apply(_pre, hs)
                if _need_mid:
                    self._resid_pre[_layer] = hs
                if args:
                    return (hs,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hs
                return args, kwargs

            self._handles.append(block.register_forward_pre_hook(block_pre, with_kwargs=True))

        if attention is not None:
            cfg = self.owner.cfg
            n_heads, d_head = cfg.n_heads, cfg.d_head

            def z_pre(module, args, kwargs, *, _hook=attention):
                z = args[0] if args else kwargs["hidden_states"]
                b, s = z.shape[0], z.shape[1]
                z = self._apply(_hook, z.view(b, s, n_heads, d_head)).reshape(b, s, -1)
                if args:
                    return (z,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = z
                return args, kwargs

            self._handles.append(
                self.family.z_proj(block).register_forward_pre_hook(z_pre, with_kwargs=True)
            )

        if attn_out is not None or resid_mid is not None:
            def attn_fwd(module, args, kwargs, output, *, _layer=layer, _attn_out=attn_out, _mid=resid_mid):
                is_tuple = isinstance(output, tuple)
                out0 = output[0] if is_tuple else output
                if _attn_out is not None:
                    out0 = self._apply(_attn_out, out0)
                if _mid is not None:
                    residual = self._resid_pre.pop(_layer)
                    h_mid = residual + out0
                    # attn_out' = edited(resid_mid) − residual: the block's own residual add
                    # then reproduces the edited resid_mid exactly (steer AND ablate).
                    out0 = out0 + (self._apply(_mid, h_mid) - h_mid)
                return (out0,) + tuple(output[1:]) if is_tuple else out0

            self._handles.append(
                self.family.attn(block).register_forward_hook(attn_fwd, with_kwargs=True)
            )

        if mlp is not None:
            def mlp_pre(module, args, kwargs, *, _hook=mlp):
                x = args[0] if args else kwargs["hidden_states"]
                x = self._apply(_hook, x)
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = x
                return args, kwargs

            self._handles.append(
                self.family.mlp_in(block).register_forward_pre_hook(mlp_pre, with_kwargs=True)
            )

        if resid_post is not None:
            def block_fwd(module, args, kwargs, output, *, _hook=resid_post):
                is_tuple = isinstance(output, tuple)
                out0 = output[0] if is_tuple else output
                out0 = self._apply(_hook, out0)
                return (out0,) + tuple(output[1:]) if is_tuple else out0

            self._handles.append(block.register_forward_hook(block_fwd, with_kwargs=True))


__all__ = ["HFHookAdapter", "load_hf_model"]

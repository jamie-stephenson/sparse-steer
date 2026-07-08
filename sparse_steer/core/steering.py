"""TransformerLens-based steering.

A single :class:`SteeringModel` wraps a ``HookedTransformer`` and injects
learned corrections at named hook points.

The model carries ONE inference engine at a time, selected by :meth:`SteeringModel.set_backend`:

- ``"tl"`` (default) — the instrumented TransformerLens ``HookedTransformer``; the correctness
  oracle and the only engine training/extraction run on.
- ``"hf"`` — a plain HF ``AutoModelForCausalLM`` (sdpa attention, ~2-5x faster eval) with the SAME
  learned steering state (the :class:`SteeringHook` modules) applied via HF forward-hooks
  (see ``core/hf_backend.py``).

Switching frees the current engine BEFORE loading the other (peak memory = max, not sum); both
engines load from the same HF checkpoint cache via the load spec recorded by ``from_pretrained``.
"""

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
import transformer_lens.weight_processing as _tl_weight_processing
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


def _install_noop_process_weights_patch() -> None:
    """Skip TransformerLens's fp32 upcast when no weight processing is requested.

    ``ProcessWeights.process_weights`` unconditionally upcasts the *entire* state
    dict to float32 (for fold/center numerical stability) before checking the
    fold/center flags — so ``from_pretrained_no_processing`` (all flags False)
    still transiently doubles the model to fp32. Alongside the source ``hf_model``
    that is ~48 GB for an 8B model, which OOMs both a 44 GB GPU and a 47 GB
    container cgroup (the death that gave no traceback). When every processing
    flag is False the upcast is a pure no-op (nothing folds), so return the dict
    untouched — the peak drops to hf_model + fp16 dict ≈ 32 GB, which fits.
    """
    PW = _tl_weight_processing.ProcessWeights
    if getattr(PW.process_weights, "_noop_patched", False):
        return
    _orig = PW.process_weights

    def process_weights(
        state_dict,
        cfg,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
        adapter=None,
    ):
        if not (
            fold_ln
            or center_writing_weights
            or center_unembed
            or fold_value_biases
            or refactor_factored_attn_matrices
        ):
            return state_dict  # no processing requested → skip the fp32 upcast
        return _orig(
            state_dict,
            cfg,
            fold_ln,
            center_writing_weights,
            center_unembed,
            fold_value_biases,
            refactor_factored_attn_matrices,
            adapter,
        )

    process_weights._noop_patched = True
    PW.process_weights = staticmethod(process_weights)


_install_noop_process_weights_patch()

Component = Literal[
    "attention", "attn_out", "mlp", "resid_pre", "resid_mid", "resid_post"
]

COMPONENT_HOOK: dict[Component, str] = {
    "attention": "blocks.{i}.attn.hook_z",
    # attn_out = the attention block's residual contribution (post-W_O, d_model). SafeSteer's
    # "attention activations" (Eq 1/2) are probably this, not the per-head hook_z.
    "attn_out": "blocks.{i}.hook_attn_out",
    "mlp": "blocks.{i}.mlp.hook_post",
    "resid_pre": "blocks.{i}.hook_resid_pre",  # block input (Arditi reads/ablates directions here)
    "resid_mid": "blocks.{i}.hook_resid_mid",
    "resid_post": "blocks.{i}.hook_resid_post",
}

_RESID_COMPONENTS = frozenset({"resid_pre", "resid_mid", "resid_post"})


# ── Gate hyperparameters ──────────────────────────────────────────────


@dataclass
class HardConcreteConfig:
    """Hyperparameters for the Hard-Concrete gate distribution."""

    temperature: float = 0.33
    stretch_limits: list[float] = field(default_factory=lambda: [-0.1, 1.1])
    eps: float = 1e-6
    eval_threshold: float = 1e-2
    init_log_alpha: float = 0.0


# ── Per-hook steering module ──────────────────────────────────────────


class SteeringHook(nn.Module):
    """Holds the steering vector, optional gate, and scale for one hook point."""

    def __init__(
        self,
        vector_shape: tuple[int, ...],
        *,
        gate_config: HardConcreteConfig | None = None,
        learn_scale: bool = False,
        init_raw_scale: float = 0.0,
        shared_raw_scale: nn.Parameter | None = None,
        steering_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.enabled = True
        # Optional ``(batch, seq)`` bool mask restricting where the correction is
        # added (set per-forward via ``SteeringModel.steer_positions``); ``None``
        # steers every position (the default).
        self.pos_mask: Tensor | None = None
        # Every steering tensor (the vector v, gate log_alpha, scale raw_scale, ablation
        # normaliser) lives in ``steering_dtype`` — the single knob for the whole steering
        # math. float32 (default) = stable; float16 = the old coupled-precision regime. The
        # correction is cast to the base activation dtype at apply time, so steering_dtype is
        # independent of the base ``model_dtype``. (``float(...)`` also guards integer config
        # overrides like ``init_raw_scale=10`` from creating a long tensor softplus can't handle.)
        self.steering_dtype = steering_dtype
        self.register_buffer("steering_vectors", torch.zeros(vector_shape, dtype=steering_dtype))
        num_gates = vector_shape[0] if len(vector_shape) > 1 else 1
        # Per-site ablation normaliser (mean |activation·v̂| over the steered
        # positions). Dividing the ablation strength by it equalises the gate
        # gradient scale across sites whose residual norm grows with depth, so
        # the learned gates select on CE-benefit rather than activation norm.
        # Default 1.0 leaves behaviour unchanged; ``set_proj_act_norms`` fills it in.
        self.register_buffer("proj_act_norm", torch.ones(num_gates, dtype=steering_dtype))
        # Hold the shared scale in a tuple so nn.Module does not re-register it
        # as a parameter of every hook (it is owned once by the SteeringModel).
        self._shared_holder = (shared_raw_scale,)

        self.gate_config = gate_config
        if gate_config is not None:
            self.log_alpha = nn.Parameter(
                torch.full((num_gates,), float(gate_config.init_log_alpha), dtype=steering_dtype)
            )
        else:
            self.register_parameter("log_alpha", None)

        if shared_raw_scale is not None:
            self.register_parameter("raw_scale", None)
        elif learn_scale:
            self.raw_scale = nn.Parameter(
                torch.full((num_gates,), float(init_raw_scale), dtype=steering_dtype)
            )
        else:
            self.register_buffer(
                "raw_scale", torch.full((num_gates,), float(init_raw_scale), dtype=steering_dtype)
            )

    @property
    def _shared_raw_scale(self) -> nn.Parameter | None:
        return self._shared_holder[0]

    # ── Gate / scale ──────────────────────────────────────────────────

    def _hard_concrete(self) -> Tensor:
        cfg = self.gate_config
        low, high = cfg.stretch_limits
        if self.training:
            noise = torch.rand_like(self.log_alpha)
            noise = noise.mul(1.0 - 2.0 * cfg.eps).add(cfg.eps)
            concrete = torch.sigmoid(
                (noise.log() - torch.log1p(-noise) + self.log_alpha) / cfg.temperature
            )
        else:
            concrete = torch.sigmoid(self.log_alpha)
        stretched = concrete * (high - low) + low
        return stretched.clamp(0.0, 1.0)

    def l0_penalty(self) -> Tensor:
        """Expected number of active gates (differentiable L0 surrogate)."""
        if self.log_alpha is None:
            return torch.tensor(0.0, device=self.steering_vectors.device)
        cfg = self.gate_config
        low, high = cfg.stretch_limits
        return torch.sigmoid(
            self.log_alpha - cfg.temperature * math.log(-low / high)
        ).sum()

    def _gate_weights(self) -> Tensor | float:
        if self.log_alpha is None:
            return 1.0
        weights = self._hard_concrete()
        if not self.training and self.gate_config.eval_threshold > 0.0:
            weights = torch.where(
                weights >= self.gate_config.eval_threshold,
                weights,
                torch.zeros_like(weights),
            )
        return weights

    def _scale_weights(self) -> Tensor:
        if self._shared_raw_scale is not None:
            return F.softplus(self._shared_raw_scale)
        return F.softplus(self.raw_scale)

    # ── Correction ────────────────────────────────────────────────────

    def _correction(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        steering = self.steering_vectors.to(device=device, dtype=dtype)
        weight = self._gate_weights() * self._scale_weights()
        if isinstance(weight, Tensor) and steering.ndim > 1:
            weight = weight.unsqueeze(-1)
        return steering * weight

    def _add_delta(self, activation: Tensor, delta: Tensor) -> Tensor:
        """Add ``delta`` to *activation*, confined to ``pos_mask`` positions if set.

        ``delta`` is cast to the activation dtype (it is computed in float32 for
        stable optimisation over a possibly-fp16 base model). When ``pos_mask``
        (a ``(batch, seq)`` bool tensor) is set the delta is zeroed outside those
        positions, so the intervention can be confined to e.g. prompt tokens only.
        """
        delta = delta.to(activation.dtype)
        if self.pos_mask is not None:
            # (batch, seq) → (batch, seq, 1, …) to broadcast over feature dims.
            mask = self.pos_mask.to(device=activation.device, dtype=activation.dtype)
            mask = mask.reshape(mask.shape + (1,) * (activation.ndim - mask.ndim))
            delta = delta * mask
        return activation + delta

    @property
    def _compute_dtype(self) -> torch.dtype:
        """Dtype the steering correction (and its gradient) is computed in — the single
        ``steering_dtype`` knob (the vector, gate, and scale params all share it)."""
        return self.steering_dtype

    def steer(self, activation: Tensor, hook=None) -> Tensor:
        """TransformerLens hook: additive steering — ``activation + scale·gate·v``.

        The correction is a fixed vector (independent of the activation),
        broadcast over all leading (batch, pos, …) dimensions.
        """
        if not self.enabled:
            return activation
        correction = self._correction(self._compute_dtype, activation.device)
        lead = activation.ndim - correction.ndim
        return self._add_delta(activation, correction.reshape((1,) * lead + correction.shape))

    def ablate(self, activation: Tensor, hook=None) -> Tensor:
        """TransformerLens hook: geometric (projection) ablation.

        Removes the activation's component along the **unit** direction ``v̂``
        (the normalized steering vector), scaled by ``scale·gate``::

            activation - scale·gate·(activation·v̂)·v̂

        Unlike :meth:`steer`, the correction depends on the activation — each
        position (and head) is pushed by its own projection ``activation·v̂``.
        ``scale·gate = 1`` is exact orthogonal-projection ablation (α=1); larger
        over-ablates. ``steering_vectors`` is already the unit direction ``v̂``
        (``SteeringModel.set_all_vectors`` normalises it for ablate), so there is
        no per-forward renormalisation.
        """
        if not self.enabled:
            return activation
        v_hat = self.steering_vectors.to(device=activation.device, dtype=self.steering_dtype)
        lead = activation.ndim - v_hat.ndim
        v_hat = v_hat.reshape((1,) * lead + v_hat.shape)
        coef = (activation.to(self.steering_dtype) * v_hat).sum(dim=-1, keepdim=True)  # activation·v̂
        weight = self._gate_weights() * self._scale_weights() / self.proj_act_norm
        if isinstance(weight, Tensor):
            # (num_gates,) → (num_gates, 1): broadcasts over coef's per-head axis
            # for attention, or the trailing singleton for residual/mlp.
            weight = weight.reshape(weight.shape + (1,))
        return self._add_delta(activation, -(weight * coef) * v_hat)

    def effective_weight(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Combined gate * scale weights as a tensor (for visualisation)."""
        weight = self._gate_weights() * self._scale_weights()
        if isinstance(weight, (int, float)):
            return torch.tensor(weight, dtype=dtype, device=device)
        return weight.to(dtype=dtype, device=device)

    def set_steering_vectors(self, vectors: Tensor) -> None:
        expected = self.steering_vectors.shape
        if vectors.shape != expected:
            raise ValueError(
                f"Steering vectors must have shape {tuple(expected)}, "
                f"got {tuple(vectors.shape)}."
            )
        self.steering_vectors.copy_(
            vectors.to(
                device=self.steering_vectors.device,
                dtype=self.steering_vectors.dtype,
            )
        )


# ── Model loading ─────────────────────────────────────────────────────


def load_hooked_transformer(
    model_name: str,
    *,
    architecture_name: str | None = None,
    lora_adapter: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    process_weights: bool = True,
) -> HookedTransformer:
    """Load a ``HookedTransformer`` for ``model_name``.

    With ``lora_adapter``, merge the adapter into the ``model_name`` base
    (TransformerLens has no native PEFT support) and hand the merged weights in
    via ``hf_model=``. ``model_name`` names the base weights the adapter is applied to
    and, by default, doubles as the architecture for TL's config.

    ``architecture_name`` splits "which architecture" from "whose weights" for
    checkpoints TL doesn't know by name (e.g. the Cadenza sleeper: dolphin-2.9-llama3-8b
    weights on the Meta-Llama-3-8B architecture). TL reads its config for
    ``architecture_name`` while the state dict AND tokenizer come from ``model_name``
    (+ merged ``lora_adapter``) — nothing is fetched from the architecture repo (which
    may be gated, e.g. meta-llama).

    With ``process_weights=False``, skip TransformerLens's weight processing
    (LayerNorm folding, weight centering, …). That processing is function-preserving
    (hooks read the same residual stream), but at reduced precision it builds a third
    full-size state dict (~3×13GB for 7B) which swaps/OOMs. It is also the *faithful*
    choice when reproducing a paper that runs the raw HF model with no TL transforms
    (e.g. SafeSteer): the activations ω is built from then match the unprocessed model.
    """
    tl_name = architecture_name or model_name
    split_source = architecture_name is not None and architecture_name != model_name
    tl_kwargs: dict = {"device": device, "dtype": dtype}
    if split_source:
        tl_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    if lora_adapter is None:
        hf_model = None
        if model_name.startswith("Qwen/Qwen-"):
            if int(transformers.__version__.split(".")[0]) >= 5:
                # Weights and tokenizer load fine but the forward pass silently
                # produces garbage (verified 2026-06-10: good on 4.49.0, broken on 5.9).
                raise RuntimeError(
                    f"transformers {transformers.__version__} silently breaks Qwen-1.0 "
                    "inference. Run this experiment with the legacy overlay:\n"
                    '  uv run --with "transformers==4.49.0" python run.py ...'
                )
            # Left to its own devices TL loads Qwen-1.0 weights in fp32 and the remote
            # code auto-casts to bf16, transiently holding both copies (~42 GB); with
            # TL's conversion dict on top this OOMs a 50 GB container. Pre-load once in
            # the target dtype, using Qwen's own precision flag to disable the auto-cast.
            qwen_flag = {
                torch.float16: "fp16",
                torch.bfloat16: "bf16",
                torch.float32: "fp32",
            }[dtype]
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                **{qwen_flag: True},
            )
            # Skip TL weight processing: it builds a third full-size state dict
            # (hf + converted + processed ≈ 46 GB for 7B) which OOMs a 50 GB
            # container, and TL itself advises no_processing at reduced precision.
            # Processing is function-preserving, so resid-stream hooks are unaffected.
            return HookedTransformer.from_pretrained_no_processing(
                model_name, hf_model=hf_model, device=device, dtype=dtype
            )
        if split_source:
            # weights source ≠ architecture: pre-load the state dict ourselves so TL
            # doesn't fetch tl_name's weights. Load straight onto `device`: kept on
            # CPU, TL would transiently hold three full model copies there — hf_model
            # + its converted state dict + the fresh HookedTransformer params (~48 GB
            # for 8B fp16) — and overflow a ~47 GB container cgroup, dying with a bare
            # SIGKILL (no traceback). Llama-3's 128k-vocab embed/unembed is exactly
            # what tips 8B over where the same-shape 7B path fits. On the GPU only
            # hf_model + the converted dict coexist (~32 GB, fits a 48 GB card) and
            # CPU holds just the HT params before they move to device.
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map={"": device} if device not in (None, "cpu") else None,
            )
        if not process_weights:
            # hf_model is None here (unless split_source) → TL fetches tl_name's weights itself.
            return HookedTransformer.from_pretrained_no_processing(
                tl_name, hf_model=hf_model, **tl_kwargs
            )
        return HookedTransformer.from_pretrained(tl_name, hf_model=hf_model, **tl_kwargs)
    from peft import PeftModel

    print(f"Merging LoRA adapter '{lora_adapter}' into base '{model_name}'...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, lora_adapter).merge_and_unload().eval()
    if not process_weights:
        return HookedTransformer.from_pretrained_no_processing(
            tl_name, hf_model=merged, **tl_kwargs
        )
    return HookedTransformer.from_pretrained(tl_name, hf_model=merged, **tl_kwargs)


# ── Steering model ────────────────────────────────────────────────────


class SteeringModel(nn.Module):
    """A ``HookedTransformer`` with learnable steering hooks attached.

    Exposes a HuggingFace-style ``forward(input_ids, attention_mask, labels)``
    returning a ``CausalLMOutputWithPast`` so the eval and LoRA-free training
    code can treat it like any causal LM.

    Two inference engines share the ONE steering state (see the module docstring):
    ``backend == "tl"`` (default; training/extraction) or ``backend == "hf"`` (fast eval),
    switched with :meth:`set_backend`. ``.tl`` / ``.hf`` access the current engine and raise
    a clear error when the other backend is active.
    """

    def __init__(
        self,
        tl: HookedTransformer,
        *,
        steering_layer_ids: list[int],
        steering_components: list[Component],
        gate_config: HardConcreteConfig | None = None,
        learn_scale: bool = False,
        shared_scale: bool = False,
        init_raw_scale: float = 0.0,
        intervention: str = "steer",
        steering_dtype: torch.dtype = torch.float32,
        load_spec: dict | None = None,
    ) -> None:
        super().__init__()
        if intervention not in ("steer", "ablate"):
            raise ValueError(f"intervention must be 'steer' or 'ablate', got {intervention!r}")
        # The engine is stored under one name so exactly ONE is ever resident; ``tl``/``hf``
        # are properties over it. ``_load_spec`` (recorded by from_pretrained) lets
        # ``set_backend`` (re)build either engine from the same HF checkpoint cache.
        self._engine = tl
        self._backend = "tl"
        self._load_spec = load_spec
        self._hf_adapter = None  # core.hf_backend.HFHookAdapter when backend == "hf"
        # Config + tokenizer survive engine swaps (the HookedTransformerConfig is a plain
        # dataclass independent of the weights; keeping it costs nothing).
        self._tl_cfg = tl.cfg
        self._tokenizer = tl.tokenizer
        self.gate_config = gate_config
        self.learn_scale = learn_scale
        self.shared_scale = shared_scale
        self.init_raw_scale = init_raw_scale
        self.intervention = intervention
        self.steering_dtype = steering_dtype
        self.steering_layer_ids = list(steering_layer_ids)
        self.steering_components = list(steering_components)

        ref = next(self._engine.parameters())
        # Steering params live in ``steering_dtype`` (float32 default = stable, even on an fp16
        # base model; the correction is cast to the activation dtype at apply time).
        if shared_scale:
            self._shared_raw_scale = nn.Parameter(
                torch.full((1,), init_raw_scale, device=ref.device, dtype=steering_dtype)
            )
        else:
            self._shared_raw_scale = None

        self.hooks = nn.ModuleDict()
        self._attach_hooks()

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        architecture_name: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        steering_dtype: torch.dtype = torch.float32,
        lora_adapter: str | None = None,
        steering_layer_ids: list[int] | None = None,
        steering_components: Sequence[Component] = ("attention",),
        gate_config: HardConcreteConfig | None = None,
        learn_scale: bool = False,
        shared_scale: bool = False,
        init_raw_scale: float = 0.0,
        intervention: str = "steer",
        process_weights: bool = True,
    ) -> "SteeringModel":
        # ``dtype`` is the base model (``model_dtype``); ``steering_dtype`` is the steering math.
        tl = load_hooked_transformer(
            model_name, architecture_name=architecture_name, lora_adapter=lora_adapter,
            device=device, dtype=dtype, process_weights=process_weights,
        )
        if steering_layer_ids is None:
            steering_layer_ids = list(range(tl.cfg.n_layers))
        # Record the load recipe so set_backend can (re)build either engine on demand from
        # the SAME HF checkpoint cache (no disk duplication).
        load_spec = dict(
            model_name=model_name,
            architecture_name=architecture_name,
            lora_adapter=lora_adapter,
            device=device,
            dtype=dtype,
            process_weights=process_weights,
        )
        return cls(
            tl,
            steering_layer_ids=steering_layer_ids,
            steering_components=list(steering_components),
            gate_config=gate_config,
            learn_scale=learn_scale,
            shared_scale=shared_scale,
            init_raw_scale=init_raw_scale,
            intervention=intervention,
            steering_dtype=steering_dtype,
            load_spec=load_spec,
        )

    def _vector_shape(self, component: Component) -> tuple[int, ...]:
        c = self._tl_cfg
        if component == "attention":
            return (c.n_heads, c.d_head)
        if component == "attn_out":
            return (c.d_model,)  # post-W_O attention output (residual contribution)
        if component == "mlp":
            return (c.d_mlp,)
        if component in _RESID_COMPONENTS:
            return (c.d_model,)
        raise ValueError(f"Unknown component: {component!r}")

    def _attach_hooks(self) -> None:
        ref = next(self._engine.parameters())
        for i in self.steering_layer_ids:
            for component in self.steering_components:
                hook = SteeringHook(
                    self._vector_shape(component),
                    gate_config=self.gate_config,
                    learn_scale=self.learn_scale,
                    init_raw_scale=self.init_raw_scale,
                    shared_raw_scale=self._shared_raw_scale,
                    steering_dtype=self.steering_dtype,
                )
                # Move to the model's device (params/buffers stay in steering_dtype).
                hook.to(device=ref.device)
                self.hooks[f"{component}_{i}"] = hook
        self._wire_tl_hooks()

    def _wire_tl_hooks(self) -> None:
        """Add the permanent TransformerLens hooks for every SteeringHook (init + tl rebuild)."""
        for component, i, hook in self.iter_hooks():
            name = COMPONENT_HOOK[component].format(i=i)
            apply_fn = hook.ablate if self.intervention == "ablate" else hook.steer
            self._engine.add_hook(name, apply_fn, is_permanent=True)

    # ── Backend (engine) management ───────────────────────────────────

    @property
    def backend(self) -> str:
        """Current inference engine: ``"tl"`` (TransformerLens) or ``"hf"`` (native HF, sdpa)."""
        return self._backend

    @property
    def tl(self) -> HookedTransformer:
        if self._backend != "tl":
            raise RuntimeError("backend is 'hf'; call set_backend('tl') to use the TL engine")
        return self._engine

    @property
    def hf(self):
        if self._backend != "hf":
            raise RuntimeError("backend is 'tl'; call set_backend('hf') to use the HF engine")
        return self._engine

    def set_backend(self, name: str) -> None:
        """Switch the inference engine to ``name`` ("tl" | "hf"). Idempotent.

        Discipline: the current engine is freed (``del`` + accelerator cache flush) BEFORE the
        other is loaded, so peak memory is the max of the two engines, never their sum. Both
        engines load from the same on-disk HF checkpoint cache via the recorded load spec.
        The steering state (the :class:`SteeringHook` modules) is engine-independent and is
        re-wired onto whichever engine is loaded.
        """
        if name not in ("tl", "hf"):
            raise ValueError(f"backend must be 'tl' or 'hf', got {name!r}")
        if name == self._backend:
            return
        if self._load_spec is None:
            raise RuntimeError(
                "This SteeringModel was built from an in-memory HookedTransformer (no load "
                "spec); backend switching requires construction via from_pretrained."
            )
        from sparse_steer.utils.memory import free_model_memory

        # 1) Free the current engine FIRST (never both resident).
        if self._hf_adapter is not None:
            self._hf_adapter.remove()
            self._hf_adapter = None
        engine = self._engine
        self._engine = None
        del engine
        free_model_memory()

        # 2) Load the requested engine from the recorded spec and re-wire the steering state.
        spec = self._load_spec
        if name == "tl":
            engine = load_hooked_transformer(
                spec["model_name"],
                architecture_name=spec["architecture_name"],
                lora_adapter=spec["lora_adapter"],
                device=spec["device"],
                dtype=spec["dtype"],
                process_weights=spec["process_weights"],
            )
            self._engine = engine
            self._backend = "tl"
            self._wire_tl_hooks()
        else:
            from .hf_backend import HFHookAdapter, load_hf_model

            engine = load_hf_model(
                spec["model_name"],
                lora_adapter=spec["lora_adapter"],
                device=spec["device"],
                dtype=spec["dtype"],
            )
            self._engine = engine
            self._backend = "hf"
            self._hf_adapter = HFHookAdapter(self)
        self._engine.train(self.training)

    # ── HF-style surface ──────────────────────────────────────────────

    @property
    def cfg(self):
        return self._tl_cfg

    @property
    def device(self) -> torch.device:
        return next(self._engine.parameters()).device

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **_: object,
    ) -> CausalLMOutputWithPast:
        """Return logits only.

        The training objective is task-specific (``TaskSpec.loss``) and is applied
        to these logits in the training loop, which also adds the steering-gate L0
        penalty (``l0_penalty``). The model itself carries no task loss.
        """
        if self._backend == "hf":
            # Position ids from the attention mask (cumsum over real tokens), matching TL's
            # get_offset_position_ids — a plain HF forward would otherwise use arange, which
            # is wrong for left-padded batches (e.g. decision_logprobs / generation prompts).
            position_ids = None
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
            out = self._engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            return CausalLMOutputWithPast(loss=None, logits=out.logits)
        logits = self.tl(
            input_ids,
            attention_mask=attention_mask,
            return_type="logits",
            prepend_bos=False,
        )
        return CausalLMOutputWithPast(loss=None, logits=logits)

    def generate(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kw):
        if self._backend == "hf":
            # Steering fires via the HF forward-hooks inside model.generate; a position mask
            # set with steer_positions is interpreted per absolute position across KV-cache
            # decode steps (see hf_backend.HFHookAdapter).
            return self._engine.generate(
                input_ids, attention_mask=attention_mask, **kw
            )
        return self.tl.generate(input_ids, **kw)

    # ── Steering vectors ──────────────────────────────────────────────

    def iter_hooks(self) -> Iterator[tuple[Component, int, SteeringHook]]:
        for key, hook in self.hooks.items():
            component, _, layer = key.rpartition("_")
            yield component, int(layer), hook

    def set_all_vectors(
        self, vectors: dict[str, Tensor], *, normalize: bool = False
    ) -> None:
        """Apply per-layer steering vectors for each steered component."""
        expected = set(self.steering_components)
        provided = set(vectors)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing expected components: {missing}")
            if unexpected:
                details.append(f"unexpected components: {unexpected}")
            raise ValueError("Invalid steering vector components; " + "; ".join(details))

        # Ablation projects onto the unit direction, so its vectors must be
        # normalised; storing them unit here means the hook never renormalises.
        if normalize or self.intervention == "ablate":
            vectors = {k: F.normalize(v, dim=-1) for k, v in vectors.items()}

        for component, tensor in vectors.items():
            for i in self.steering_layer_ids:
                key = f"{component}_{i}"
                if key in self.hooks:
                    self.hooks[key].set_steering_vectors(tensor[i])
        if self._hf_adapter is not None:
            self._hf_adapter.rewire()  # site activity may have changed

    @contextmanager
    def steering_disabled(self):
        for _, _, hook in self.iter_hooks():
            hook.enabled = False
        try:
            yield
        finally:
            for _, _, hook in self.iter_hooks():
                hook.enabled = True

    @contextmanager
    def steer_positions(self, mask: Tensor | None):
        """Restrict steering to the ``True`` positions of ``mask`` (``(batch, seq)``).

        Within the block every steering hook only adds its correction at the
        masked positions; outside it (``mask=None``) steering applies everywhere,
        the default. Used to confine steering to prompt tokens during evaluation.
        With KV-cached generation the prompt-only steer therefore needs to fire
        on just the prompt forward — the steered prompt key/values are cached and
        attended to by later decode steps with steering off.
        """
        for _, _, hook in self.iter_hooks():
            hook.pos_mask = mask
        try:
            yield
        finally:
            for _, _, hook in self.iter_hooks():
                hook.pos_mask = None

    @staticmethod
    def last_token_mask(attention_mask: Tensor) -> Tensor:
        """Boolean mask selecting each row's last real token.

        Works for both left- and right-padded batches by taking the largest index whose
        attention mask is true.
        """
        valid = attention_mask.bool()
        positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        last = positions.masked_fill(~valid, -1).max(dim=1).values
        out = torch.zeros_like(valid)
        rows = torch.arange(valid.shape[0], device=valid.device)
        ok = last >= 0
        out[rows[ok], last[ok]] = True
        return out

    @contextmanager
    def steer_last_token(self, attention_mask: Tensor):
        """Restrict steering to the last real token of each row."""
        with self.steer_positions(self.last_token_mask(attention_mask)):
            yield

    @torch.no_grad()
    def set_proj_act_norms(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        pos_mask: Tensor,
    ) -> None:
        """Set each ablation hook's ``proj_act_norm`` to the mean ``|activation·v̂|``
        over the ``pos_mask`` positions (with steering disabled).

        The residual norm grows with depth, so a deep site's ablation delta (and
        thus its gate gradient) is many times larger than a shallow site's. Left
        uncorrected the L0 gates select on activation norm, not on objective
        benefit. Dividing each site's ablation by its own projection magnitude
        equalises the gradient scale across sites — a uniform, site-agnostic
        normalisation that encodes no preference for any particular site.
        """
        mask = pos_mask.reshape(-1).bool()
        sums: dict[str, Tensor] = {}

        def make(key: str, hook: SteeringHook):
            v_hat = hook.steering_vectors

            def fn(act, hook=None):
                coef = (act.to(torch.float32) * v_hat.to(act.device)).sum(-1).abs()
                flat = coef.reshape(-1, coef.shape[-1]) if coef.ndim == 3 else coef.reshape(-1, 1)
                sums[key] = flat[mask.to(flat.device)].mean(0)
                return act

            return fn

        fwd_hooks = []
        for component, layer, hook in self.iter_hooks():
            name = COMPONENT_HOOK[component].format(i=layer)
            fwd_hooks.append((name, make(f"{component}_{layer}", hook)))
        with self.steering_disabled():
            self.tl.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                return_type=None,
                fwd_hooks=fwd_hooks,
                prepend_bos=False,
            )
        for component, layer, hook in self.iter_hooks():
            val = sums[f"{component}_{layer}"].clamp_min(1e-6)
            hook.proj_act_norm.copy_(val.to(hook.proj_act_norm))

    # ── Freezing ──────────────────────────────────────────────────────

    def _unfreeze_scale(self) -> None:
        if self._shared_raw_scale is not None:
            self._shared_raw_scale.requires_grad = True
        else:
            for _, _, hook in self.iter_hooks():
                if hook.raw_scale is not None and isinstance(hook.raw_scale, nn.Parameter):
                    hook.raw_scale.requires_grad = True

    def freeze_base_model(self, freeze_raw_scale: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for _, _, hook in self.iter_hooks():
            if hook.log_alpha is not None:
                hook.log_alpha.requires_grad = True
        if not freeze_raw_scale:
            self._unfreeze_scale()

    def l0_penalty(self) -> Tensor:
        penalty = torch.tensor(0.0, device=self.device)
        for _, _, hook in self.iter_hooks():
            penalty = penalty + hook.l0_penalty()
        return penalty

    # ── Save / load ───────────────────────────────────────────────────

    def steering_state_dict(self) -> dict[str, Tensor]:
        terms = ("log_alpha", "raw_scale", "steering_vectors", "_shared_raw_scale", "proj_act_norm")
        return {
            k: v
            for k, v in self.state_dict().items()
            if any(term in k for term in terms)
        }

    def save_steering(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix:
            output_path = path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path = path / "steering.pt"
        payload = {
            "gate_config": asdict(self.gate_config) if self.gate_config else None,
            "learn_scale": self.learn_scale,
            "shared_scale": self.shared_scale,
            "init_raw_scale": self.init_raw_scale,
            "steering_layer_ids": self.steering_layer_ids,
            "steering_components": self.steering_components,
            "state_dict": self.steering_state_dict(),
        }
        torch.save(payload, output_path)
        return output_path

    def load_steering(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        load_info = self.load_state_dict(payload["state_dict"], strict=False)
        missing = [
            k
            for k in load_info.missing_keys
            if any(t in k for t in ("log_alpha", "raw_scale", "steering_vector"))
        ]
        if missing:
            raise RuntimeError(
                f"Steering checkpoint mismatch. Missing steering keys: {missing}."
            )
        if self._hf_adapter is not None:
            self._hf_adapter.rewire()  # gates/vectors changed → recompute skipped sites


__all__ = [
    "Component",
    "COMPONENT_HOOK",
    "HardConcreteConfig",
    "SteeringHook",
    "SteeringModel",
]

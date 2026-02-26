from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
)

from ..utils.hardconcrete import HardConcreteConfig, HardConcreteGateMixin


class Attention(HardConcreteGateMixin, LlamaAttention):
    def __init__(
        self, *args: Any, gate_config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._set_gate_config(gate_config)
        self.steering_enabled = True

        self.register_buffer(
            "steering_vectors",
            torch.zeros(self.num_heads, self.head_dim),
            persistent=True,
        )
        self.log_alpha = nn.Parameter(torch.zeros(self.num_heads))
        self.log_scale = nn.Parameter(torch.zeros(self.num_heads))
        self._reset_gate_parameters(self.log_alpha, self.log_scale)

        # Apply steering before o_proj so the intervention is per-head in pre-projection space.
        self._o_proj_hook_handle = self.o_proj.register_forward_pre_hook(
            self._apply_steering_before_o_proj
        )

    def set_steering_vectors(self, vectors: Tensor) -> None:
        expected = self.steering_vectors.shape
        if vectors.shape != expected:
            raise ValueError(
                f"""
                Attention steering vectors must have shape 
                {tuple(expected)}, got {tuple(vectors.shape)}."
                """
            )
        self.steering_vectors.copy_(
            vectors.to(
                device=self.steering_vectors.device, 
                dtype=self.steering_vectors.dtype
            )
        )

    def _apply_steering_before_o_proj(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Optional[tuple[Any, ...]]:
        if not self.steering_enabled or not inputs:
            return None

        # (batch_size, seq_len, hidden_size)
        attn_output = inputs[0]
        assert isinstance(attn_output, torch.Tensor), (
            "o_proj hook saw a non-tensor input"
        )
        expected_hidden_size = self.num_heads * self.head_dim
        assert attn_output.shape[-1] != expected_hidden_size, (
            "o_proj received an unexpected tensor shape"
        )

        # (num_heads,)
        gate = self._scaled_gate(
            self.log_alpha,
            self.log_scale,
            dtype=attn_output.dtype,
            device=attn_output.device,
        )

        # (num_heads, head_dim)
        steering = self.steering_vectors.to(
            device=attn_output.device, dtype=attn_output.dtype
        )

        # (1, 1, hidden_size)
        correction = (steering * gate.unsqueeze(-1)).reshape(1, 1, expected_hidden_size)
        steered_output = attn_output + correction

        return (steered_output, *inputs[1:])


def _copy_module_state(source: nn.Module, target: nn.Module) -> None:
    source_param = next(source.parameters(), None)
    if source_param is not None:
        target.to(device=source_param.device, dtype=source_param.dtype)
    target.load_state_dict(source.state_dict(), strict=False)


def _build_attention_module(
    source: LlamaAttention,
    gate_config: Optional[dict[str, Any]],
    model_config: Any,
) -> Attention:
    if isinstance(source, Attention):
        source._set_gate_config(gate_config)
        return source

    source_config = getattr(source, "config", model_config)
    kwargs: dict[str, Any] = {}
    if hasattr(source, "layer_idx"):
        kwargs["layer_idx"] = getattr(source, "layer_idx")
    try:
        upgraded = Attention(source_config, gate_config=gate_config, **kwargs)
    except TypeError:
        kwargs.pop("layer_idx", None)
        upgraded = Attention(source_config, gate_config=gate_config, **kwargs)
    _copy_module_state(source, upgraded)
    return upgraded


class SparseSteeringLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: Any) -> None:
        sparse_cfg = HardConcreteConfig.from_dict(
            getattr(config, "sparse_steering", None)
        )
        config.sparse_steering = sparse_cfg.to_dict()
        super().__init__(config)
        self._upgrade_llama_modules()

    def _upgrade_llama_modules(self) -> None:
        gate_config = getattr(self.config, "sparse_steering", None)
        for layer in self.model.layers:
            layer.self_attn = _build_attention_module(
                layer.self_attn, gate_config, self.config
            )
            layer.mlp = _build_mlp_module(layer.mlp, gate_config, self.config)

    def set_steering_enabled(self, enabled: bool) -> None:
        for layer in self.model.layers:
            layer.self_attn.steering_enabled = enabled
            layer.mlp.steering_enabled = enabled

    def set_attention_steering(self, layer_idx: int, vectors: Tensor) -> None:
        self.model.layers[layer_idx].self_attn.set_steering_vectors(vectors)

    def set_all_attention_steering(self, vectors: Tensor) -> None:
        expected_shape = (
            len(self.model.layers),
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        if vectors.shape != expected_shape:
            raise ValueError(
                f"All attention steering vectors must have shape {expected_shape}, got {tuple(vectors.shape)}."
            )
        for idx, layer in enumerate(self.model.layers):
            layer.self_attn.set_steering_vectors(vectors[idx])

    def set_all_mlp_steering(self, vectors: Tensor) -> None:
        expected_shape = (len(self.model.layers), self.config.intermediate_size)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"All MLP steering vectors must have shape {expected_shape}, got {tuple(vectors.shape)}."
            )
        for idx, layer in enumerate(self.model.layers):
            layer.mlp.set_steering_vector(vectors[idx])

    def freeze_base_model(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            layer.self_attn.log_alpha.requires_grad = True
            layer.self_attn.log_scale.requires_grad = True
            layer.mlp.log_alpha.requires_grad = True
            layer.mlp.log_scale.requires_grad = True

    def sparse_steering_state_dict(self) -> dict[str, Tensor]:
        steering_terms = (
            "log_alpha",
            "log_scale",
            "steering_vector",
            "steering_vectors",
        )
        return {
            k: v
            for k, v in self.state_dict().items()
            if any(term in k for term in steering_terms)
        }

    def save_sparse_steering(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix:
            output_path = path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path = path / "sparse_steering.pt"
        payload = {
            "config": getattr(self.config, "sparse_steering", {}),
            "state_dict": self.sparse_steering_state_dict(),
        }
        torch.save(payload, output_path)
        return output_path

    def load_sparse_steering(self, path: str | Path, strict: bool = True) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        config = payload.get("config")
        if config:
            self.config.sparse_steering = config
            self._upgrade_llama_modules()
        state_dict = payload["state_dict"]
        load_info = self.load_state_dict(state_dict, strict=False)
        if strict and (load_info.unexpected_keys or load_info.missing_keys):
            missing = [
                k
                for k in load_info.missing_keys
                if any(
                    term in k for term in ("log_alpha", "log_scale", "steering_vector")
                )
            ]
            if missing or load_info.unexpected_keys:
                raise RuntimeError(
                    "Sparse steering checkpoint mismatch. "
                    f"Missing steering keys: {missing}. Unexpected keys: {load_info.unexpected_keys}."
                )


__all__ = [
    "Attention",
    "HardConcreteConfig",
    "SparseSteeringLlamaForCausalLM",
]

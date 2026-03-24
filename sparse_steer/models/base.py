from abc import ABC, abstractmethod
from typing import Literal

from torch import nn

Component = Literal["attention", "mlp", "residual"]


class BaseModelLayout(ABC):
    """Abstract layout that maps steering to architecture-specific attributes."""

    @abstractmethod
    def get_layers(self) -> nn.ModuleList: ...

    @abstractmethod
    def get_attention(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def get_mlp(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_vector_shape(
        self, component: Component, layer: nn.Module
    ) -> tuple[int, ...]: ...

    @abstractmethod
    def _get_output_proj(self, module: nn.Module) -> nn.Module: ...


__all__ = [
    "BaseModelLayout",
    "Component",
]

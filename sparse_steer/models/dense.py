from torch import Tensor, nn

from .base import BaseSteeringLM, Component, SteeringHook


class DenseSteeringHook(SteeringHook):
    """Steering hook that scales the vector by a fixed strength."""

    def __init__(self, vector_shape: tuple[int, ...], steering_strength: float = 1.0) -> None:
        super().__init__(vector_shape)
        self.steering_strength = steering_strength

    def _compute_correction(self, hidden: Tensor) -> Tensor:
        steering = self.steering_vectors.to(device=hidden.device, dtype=hidden.dtype)
        shape = [1] * (hidden.ndim - 1) + [-1]
        return (self.steering_strength * steering).reshape(shape)


class DenseSteeringLM(BaseSteeringLM):
    """Mixin that adds dense steering to a ``PreTrainedModel``."""

    def upgrade_for_steering(
        self,
        steering_strength: float,
        steering_layer_ids: list[int],
        steering_components: list[Component],
    ) -> None:
        self.steering_strength = steering_strength
        self.steering_layer_ids = steering_layer_ids
        self.steering_components = steering_components
        self._attach_steering_hooks()

    def _create_hook(self, component: Component, layer: nn.Module) -> SteeringHook:
        shape = self._get_vector_shape(component, layer)
        return DenseSteeringHook(shape, self.steering_strength)


__all__ = [
    "DenseSteeringHook",
    "DenseSteeringLM",
]

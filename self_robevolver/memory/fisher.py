"""Diagonal Fisher EMA and shared Tucker-coordinate retention."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ..model.tucker6d import PARAMETER_NAMES, Tucker6DLinear

SHARED_NAMES = ("u_out", "u_in", "core")


class FisherMemory:
    def __init__(self) -> None:
        self.anchor: dict[str, dict[str, Tensor]] = {}
        self.diagonal: dict[str, dict[str, Tensor]] = {}
        self.samples = 0
        self.smoothing = 0.9

    @staticmethod
    def _shared_slice(old: Tensor, new: Tensor) -> tuple[slice, ...]:
        if old.ndim != new.ndim or any(old_size > new_size for old_size, new_size in zip(old.shape, new.shape)):
            raise ValueError("adapter shrank or changed parameter dimensionality")
        return tuple(slice(0, extent) for extent in old.shape)

    def penalty(self, adapters: Mapping[str, Tucker6DLinear]) -> Tensor:
        if not adapters:
            raise ValueError("no adapters")
        reference = next(iter(adapters.values())).core
        result = reference.new_zeros(())
        for path, old_params in self.anchor.items():
            if path not in adapters:
                raise KeyError(f"missing adapter {path}")
            module = adapters[path]
            for name in SHARED_NAMES:
                anchor = old_params[name]
                current = getattr(module, name)
                region = self._shared_slice(anchor, current)
                fisher = self.diagonal[path][name].to(current.device)
                delta = current[region] - anchor.to(current.device)
                result = result + (fisher * delta.square()).sum()
        return result

    def consolidate(
        self,
        adapters: Mapping[str, Tucker6DLinear],
        squared_gradients: Mapping[str, Mapping[str, Tensor]],
        sample_count: int,
        smoothing: float = 0.9,
    ) -> None:
        """Apply the paper's stage-wise Fisher EMA, Eq. 18--19.

        `squared_gradients[path][name]` must be the sum over on-policy samples,
        obtained from the final policy after training, prior to clipping.
        """
        if sample_count < 1 or not 0 <= smoothing < 1 or set(squared_gradients) != set(adapters):
            raise ValueError("invalid sample count or adapter paths")
        next_anchor = dict(self.anchor)
        next_diagonal = dict(self.diagonal)
        for path, module in adapters.items():
            gradients = squared_gradients[path]
            if set(gradients) != set(PARAMETER_NAMES):
                raise ValueError(f"incomplete Fisher gradients for {path}")
            new_anchor = {}
            new_diagonal = {}
            for name in PARAMETER_NAMES:
                current = getattr(module, name).detach().cpu().float()
                incoming = gradients[name].detach().cpu().float() / sample_count
                if incoming.shape != current.shape or not torch.isfinite(incoming).all() or (incoming < 0).any():
                    raise ValueError(f"invalid Fisher tensor for {path}.{name}")
                if self.samples and path in self.diagonal:
                    old = self.diagonal[path][name]
                    old_padded = torch.zeros_like(incoming)
                    old_padded[self._shared_slice(old, old_padded)] = old
                    incoming = smoothing * old_padded + (1 - smoothing) * incoming
                else:
                    incoming = (1 - smoothing) * incoming
                new_anchor[name] = current.clone()
                new_diagonal[name] = incoming
            next_anchor[path] = new_anchor
            next_diagonal[path] = new_diagonal
        self.anchor = next_anchor
        self.diagonal = next_diagonal
        self.samples += sample_count
        self.smoothing = smoothing

    def state_dict(self) -> dict:
        return {"version": 2, "smoothing": self.smoothing,
                "samples": self.samples, "anchor": self.anchor, "diagonal": self.diagonal}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "FisherMemory":
        if state.get("version") != 2 or not 0 <= state.get("smoothing", -1) < 1:
            raise ValueError("unsupported Fisher memory")
        memory = cls()
        memory.samples = int(state["samples"])
        memory.smoothing = float(state["smoothing"])
        memory.anchor = {path: dict(values) for path, values in state["anchor"].items()}
        memory.diagonal = {path: dict(values) for path, values in state["diagonal"].items()}
        return memory

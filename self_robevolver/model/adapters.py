"""Install 6D Tucker layers in selected frozen-backbone projections."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from torch import nn

from ..task_registry import FactorRoute
from .tucker6d import Tucker6DLinear


DEFAULT_PROJECTIONS = ("q_proj", "v_proj")
_PROJECTION_PATH = re.compile(r"(?:.*\.)?([a-z_]+)")


def install_tucker_adapters(
    model: nn.Module,
    *,
    projections: Sequence[str] = DEFAULT_PROJECTIONS,
    ranks: Sequence[int] = (12, 12, 3, 6, 3, 3),
    growth: Sequence[int] = (1, 1, 1, 1, 1, 1),
    scale: float = 1.0,
) -> dict[str, Tucker6DLinear]:
    """Freeze the backbone and replace selected transformer projections in place.

    The caller must load its chosen base checkpoint first and must not use a
    PEFT wrapper. New adapter parameters remain trainable; action heads are
    intentionally outside this function.
    """
    if not projections or len(set(projections)) != len(projections):
        raise ValueError("projections must be a nonempty list without duplicates")
    if any(name not in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} for name in projections):
        raise ValueError("unsupported language-model projection")
    if any(isinstance(module, Tucker6DLinear) for module in model.modules()):
        raise ValueError("Tucker adapters are already installed")
    candidates = []
    for path, module in model.named_modules():
        match = _PROJECTION_PATH.fullmatch(path)
        if match and match.group(1) in projections:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{path} is not nn.Linear")
            candidates.append((path, module))
    if not candidates:
        raise ValueError("no selected backbone projection matched")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed = {}
    for path, linear in candidates:
        parent_path, _, attribute = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        adapter = Tucker6DLinear(linear, ranks=ranks, growth=growth, scale=scale)
        setattr(parent, attribute, adapter)
        installed[path] = adapter
    return installed


def begin_task_all(adapters: Mapping[str, Tucker6DLinear], route: FactorRoute) -> dict[str, frozenset[str]]:
    if not adapters:
        raise ValueError("no Tucker adapters")
    return {path: adapter.begin_task(route) for path, adapter in adapters.items()}


def adapter_parameters(adapters: Mapping[str, Tucker6DLinear]) -> list[nn.Parameter]:
    return [parameter for adapter in adapters.values() for parameter in adapter.adapter_parameters()]


def adapter_state(adapters: Mapping[str, Tucker6DLinear]) -> dict:
    return {path: adapter.adapter_state() for path, adapter in adapters.items()}


def restore_tucker_adapters(model: nn.Module, state: Mapping[str, dict]) -> dict[str, Tucker6DLinear]:
    if not state:
        raise ValueError("empty adapter state")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed = {}
    for path, layer_state in state.items():
        match = _PROJECTION_PATH.fullmatch(path)
        if not match:
            raise ValueError(f"invalid projection path: {path}")
        parent_path, _, attribute = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        base = getattr(parent, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is not an undecorated nn.Linear")
        adapter = Tucker6DLinear.from_adapter_state(base, layer_state)
        setattr(parent, attribute, adapter)
        installed[path] = adapter
    return installed

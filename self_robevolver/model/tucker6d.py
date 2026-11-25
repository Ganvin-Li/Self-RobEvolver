"""Six-mode TuKA: output, input, embodiment, scene, environment and style."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..task_registry import FactorRoute

FACTOR_NAMES = ("embodiment", "scene", "environment", "instruction_style")
PARAMETER_NAMES = ("u_out", "u_in", "core", "u_embodiment", "u_scene", "u_environment", "u_instruction_style")
_INFERENCE_ROUTE: ContextVar[FactorRoute | None] = ContextVar("self_robevolver_factor_route", default=None)


@contextmanager
def use_factor_route(route: FactorRoute) -> Iterator[None]:
    """Select one factor tuple for nested UniVLA linear layers during a call."""
    if not isinstance(route, FactorRoute):
        raise TypeError("route must be FactorRoute")
    token = _INFERENCE_ROUTE.set(route)
    try:
        yield
    finally:
        _INFERENCE_ROUTE.reset(token)


def _validate_ranks(values: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != 6 or any(type(value) is not int or value < 1 for value in result):
        raise ValueError(f"{label} must be six positive integers")
    return result


class Tucker6DLinear(nn.Module):
    """Frozen linear transform plus a factor-composed Tucker update.

    `begin_task` changes parameter shapes only at a task boundary. Recreate the
    optimizer immediately afterward. Use `protected_optimizer_step` to enforce frozen
    coordinates even with Adam momentum or nonzero optimizer weight decay.
    """

    def __init__(
        self,
        base: nn.Linear,
        ranks: Sequence[int] = (12, 12, 3, 6, 3, 3),
        growth: Sequence[int] = (1, 1, 1, 1, 1, 1),
        init_std: float = 0.02,
        scale: float = 1.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        self.base = base
        self.base.requires_grad_(False)
        self.ranks = _validate_ranks(ranks, "ranks")
        self.growth = _validate_ranks(growth, "growth")
        if init_std <= 0 or scale <= 0:
            raise ValueError("init_std and scale must be positive")
        self.init_std = float(init_std)
        self.scale = float(scale)
        device = base.weight.device
        out_features, in_features = base.weight.shape
        self.u_out = nn.Parameter(torch.empty(out_features, self.ranks[0], device=device, dtype=torch.float32))
        self.u_in = nn.Parameter(torch.empty(in_features, self.ranks[1], device=device, dtype=torch.float32))
        nn.init.normal_(self.u_out, std=init_std)
        nn.init.normal_(self.u_in, std=init_std)
        self.core = nn.Parameter(torch.zeros(self.ranks, device=device, dtype=torch.float32))
        for axis, rank in zip(FACTOR_NAMES, self.ranks[2:]):
            setattr(self, f"u_{axis}", nn.Parameter(torch.empty(0, rank, device=device, dtype=torch.float32)))
        self.register_buffer("core_allowed", torch.ones(self.ranks, dtype=torch.bool, device=device))
        self.category_index: dict[str, dict[str, int]] = {axis: {} for axis in FACTOR_NAMES}
        self.valid_widths: dict[str, list[int]] = {axis: [] for axis in FACTOR_NAMES}
        self.active_route: FactorRoute | None = None
        self._train_masks: dict[str, Tensor] = {}
        self._pre_task: dict[str, Tensor] = {}
        self._new_axes: frozenset[str] = frozenset()
        self._expanded = False
        self._tasks_started = 0

    def _replace(self, name: str, value: Tensor) -> None:
        setattr(self, name, nn.Parameter(value.detach().clone()))

    def _expand(self, new_ranks: tuple[int, ...]) -> None:
        old = self.ranks
        with torch.no_grad():
            for name, dim in (("u_out", 0), ("u_in", 1)):
                original = getattr(self, name)
                wider = original.new_empty(original.shape[0], new_ranks[dim])
                wider[:, : old[dim]] = original
                nn.init.normal_(wider[:, old[dim] :], std=self.init_std)
                self._replace(name, wider)
            for dim, axis in enumerate(FACTOR_NAMES, start=2):
                name = f"u_{axis}"
                original = getattr(self, name)
                wider = original.new_zeros(original.shape[0], new_ranks[dim])
                wider[:, : old[dim]] = original
                self._replace(name, wider)
            new_core = self.core.new_zeros(new_ranks)
            old_slices = tuple(slice(0, rank) for rank in old)
            new_core[old_slices] = self.core
            self._replace("core", new_core)
            allowed = torch.ones(new_ranks, dtype=torch.bool, device=new_core.device)
            allowed[old_slices] = self.core_allowed
            old_factors = tuple(slice(0, rank) for rank in old[2:])
            allowed[(slice(old[0], None), slice(None), *old_factors)] = False
            allowed[(slice(None), slice(old[1], None), *old_factors)] = False
            self.core_allowed = allowed
        self.ranks = new_ranks

    def _append_category(self, axis: str, category: str) -> None:
        if category in self.category_index[axis]:
            return
        name = f"u_{axis}"
        original = getattr(self, name)
        row = original.new_empty(1, original.shape[1])
        # Unit-scale rows keep zero-core initialization trainable.
        nn.init.normal_(row, std=1.0 / math.sqrt(row.shape[1]))
        self._replace(name, torch.cat((original, row), dim=0))
        self.category_index[axis][category] = original.shape[0]
        self.valid_widths[axis].append(original.shape[1])

    def begin_task(self, route: FactorRoute) -> frozenset[str]:
        """Select factors and apply paper Eq. 20--24 expansion if needed.

        Returns the newly encountered factor axes. The first task initializes
        rows without expanding the supplied initial rank.
        """
        if not isinstance(route, FactorRoute):
            raise TypeError("route must be FactorRoute")
        categories = route.categories()
        if any(not category for category in categories):
            raise ValueError("all factor categories must be nonempty")
        previous = {name: getattr(self, name).detach().clone() for name in PARAMETER_NAMES}
        old_ranks = self.ranks
        self._new_axes = frozenset(
            axis for axis, category in zip(FACTOR_NAMES, categories)
            if category not in self.category_index[axis]
        )
        self._expanded = bool(self._new_axes and self._tasks_started)
        if self._expanded:
            increments = (
                self.growth[0], self.growth[1],
                *(self.growth[dim] if axis in self._new_axes else 0 for dim, axis in enumerate(FACTOR_NAMES, start=2)),
            )
            self._expand(tuple(rank + extra for rank, extra in zip(self.ranks, increments)))
        for axis, category in zip(FACTOR_NAMES, categories):
            self._append_category(axis, category)
        self.active_route = route
        self._pre_task = previous
        self._train_masks = self._make_train_masks(old_ranks)
        self._tasks_started += 1
        return self._new_axes

    def _make_train_masks(self, old_ranks: tuple[int, ...]) -> dict[str, Tensor]:
        expanded = self._expanded
        masks = {}
        for name, old_width in (("u_out", old_ranks[0]), ("u_in", old_ranks[1])):
            value = getattr(self, name)
            mask = torch.ones_like(value, dtype=torch.bool)
            if expanded:
                mask[:, :old_width] = False
            masks[name] = mask
        core_mask = self.core_allowed.clone()
        if expanded:
            core_mask[tuple(slice(0, rank) for rank in old_ranks)] = False
        masks["core"] = core_mask
        assert self.active_route is not None
        for axis, category in zip(FACTOR_NAMES, self.active_route.categories()):
            name = f"u_{axis}"
            value = getattr(self, name)
            mask = torch.zeros_like(value, dtype=torch.bool)
            row = self.category_index[axis][category]
            mask[row, : self.valid_widths[axis][row]] = True
            masks[name] = mask
        return masks

    def compose_delta(self, route: FactorRoute | None = None) -> Tensor:
        compact = self._compact_core(route)
        return (self.u_out @ compact @ self.u_in.T) * self.scale

    def _compact_core(self, route: FactorRoute | None = None) -> Tensor:
        route = route or _INFERENCE_ROUTE.get() or self.active_route
        if route is None:
            raise RuntimeError("begin_task or an explicit learned route is required")
        rows = []
        for axis, category in zip(FACTOR_NAMES, route.categories()):
            if category not in self.category_index[axis]:
                raise KeyError(f"unlearned {axis} category: {category}")
            rows.append(getattr(self, f"u_{axis}")[self.category_index[axis][category]])
        compact = self.core
        for row in reversed(rows):
            compact = torch.tensordot(compact, row, dims=([-1], [0]))
        return compact

    def forward(self, x: Tensor, route: FactorRoute | None = None) -> Tensor:
        base_output = self.base(x)
        compact = self._compact_core(route).to(dtype=x.dtype)
        hidden = F.linear(x, self.u_in.T.to(dtype=x.dtype))
        hidden = F.linear(hidden, compact)
        correction = F.linear(hidden, self.u_out.to(dtype=x.dtype)) * self.scale
        return base_output + correction.to(dtype=base_output.dtype)

    def regularization(self, *, lambda_orth: float = 0.0, lambda_cons: float = 0.0) -> Tensor:
        """FKIE Eq. 13--15 for newly allocated and inherited factor rows."""
        if self.active_route is None:
            raise RuntimeError("begin_task is required")
        loss = self.core.new_zeros(())
        for axis, category in zip(FACTOR_NAMES, self.active_route.categories()):
            name = f"u_{axis}"
            row_id = self.category_index[axis][category]
            row = getattr(self, name)[row_id]
            if axis in self._new_axes:
                old_rows = self._pre_task[name]
                if lambda_orth and old_rows.shape[0]:
                    padded = F.pad(old_rows, (0, row.numel() - old_rows.shape[1]))
                    loss = loss + lambda_orth * (
                        F.normalize(row.unsqueeze(0), dim=-1)
                        @ F.normalize(padded, dim=-1).T
                    ).square().sum()
            elif lambda_cons:
                loss = loss + lambda_cons * (row - self._pre_task[name][row_id]).square().sum()
        return loss

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Convenience form for a model containing exactly one Tucker layer."""
        protected_optimizer_step((self,), optimizer)

    def _restore_structural_zeros(self) -> None:
        self.core.masked_fill_(~self.core_allowed, 0.0)
        for axis in FACTOR_NAMES:
            matrix = getattr(self, f"u_{axis}")
            for row_id, width in enumerate(self.valid_widths[axis]):
                matrix[row_id, width:] = 0.0

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(getattr(self, name) for name in PARAMETER_NAMES)

    def train_masks(self) -> dict[str, Tensor]:
        if not self._train_masks:
            raise RuntimeError("begin_task is required")
        return dict(self._train_masks)

    def adapter_state(self) -> dict:
        """Save adaptation memory without duplicating the frozen backbone."""
        return {
            "version": 1,
            "ranks": self.ranks,
            "growth": self.growth,
            "init_std": self.init_std,
            "scale": self.scale,
            "category_index": self.category_index,
            "valid_widths": self.valid_widths,
            "tasks_started": self._tasks_started,
            "core_allowed": self.core_allowed.detach().cpu(),
            "parameters": {name: getattr(self, name).detach().cpu() for name in PARAMETER_NAMES},
        }

    @classmethod
    def from_adapter_state(cls, base: nn.Linear, state: Mapping) -> "Tucker6DLinear":
        if state.get("version") != 1:
            raise ValueError("unsupported adapter state version")
        module = cls(base, state["ranks"], state["growth"], state["init_std"], state["scale"])
        device = base.weight.device
        for name in PARAMETER_NAMES:
            module._replace(name, state["parameters"][name].to(device=device, dtype=torch.float32))
        module.core_allowed = state["core_allowed"].to(device=device, dtype=torch.bool)
        module.category_index = {axis: dict(state["category_index"][axis]) for axis in FACTOR_NAMES}
        module.valid_widths = {axis: list(state["valid_widths"][axis]) for axis in FACTOR_NAMES}
        module._tasks_started = int(state["tasks_started"])
        return module


def protected_optimizer_step(
    modules: Sequence[Tucker6DLinear], optimizer: torch.optim.Optimizer
) -> None:
    """Step once while restoring frozen coordinates across all Tucker layers.

    Rebuild the optimizer after every task boundary because expansion replaces
    Parameters. Other trainable parameters may share this optimizer.
    """
    if not modules:
        raise ValueError("at least one Tucker layer is required")
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(id(parameter) not in optimizer_ids for module in modules for parameter in module.adapter_parameters()):
        raise ValueError("optimizer is stale or missing Tucker parameters; rebuild it after begin_task")
    snapshots: list[tuple[Tucker6DLinear, dict[str, Tensor]]] = []
    seen: set[int] = set()
    with torch.no_grad():
        for module in modules:
            if id(module) in seen:
                raise ValueError("duplicate Tucker layer")
            seen.add(id(module))
            if not module._train_masks:
                raise RuntimeError("begin_task is required for every Tucker layer")
            protected = {}
            for name in PARAMETER_NAMES:
                parameter = getattr(module, name)
                mask = module._train_masks[name]
                if parameter.grad is not None:
                    parameter.grad.mul_(mask)
                protected[name] = parameter.detach().clone()
            snapshots.append((module, protected))
        optimizer.step()
        for module, protected in snapshots:
            for name in PARAMETER_NAMES:
                parameter = getattr(module, name)
                mask = module._train_masks[name]
                parameter.copy_(torch.where(mask, parameter, protected[name]))
            module._restore_structural_zeros()

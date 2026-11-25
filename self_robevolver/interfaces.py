"""Backbone, executed-rollout and environment integration contracts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import torch
from torch import Tensor, nn

@dataclass
class Decision:
    context: Any
    # Executed student samples, indexed in the shared latent-action vocabulary.
    action_indices: Tensor
    # Real observations after this decision in the same executed rollout.
    future_observations: Any

@dataclass
class Trajectory:
    decisions: Sequence[Decision]
    # Q(o_k, I) in [0,1], including the final observation: len(decisions)+1.
    quality: Tensor
    # Frozen CLIP features from training observations/instructions only.
    image_features: Tensor
    text_features: Tensor

    def validate(self):
        if not self.decisions or self.quality.shape != (len(self.decisions) + 1,):
            raise ValueError("trajectory needs decisions and a terminal quality score")
        if (not self.quality.is_floating_point() or not torch.isfinite(self.quality).all()
                or ((self.quality < 0) | (self.quality > 1)).any()):
            raise ValueError("quality must contain finite floating values in [0,1]")
        if (self.image_features.ndim != 2 or self.text_features.ndim != 2
                or self.image_features.shape[0] != len(self.decisions)
                or self.text_features.shape[0] != len(self.decisions)):
            raise ValueError("retrieval features must align with the decision steps")
        for d in self.decisions:
            if (d.action_indices.ndim != 1 or not d.action_indices.numel()
                    or d.action_indices.dtype not in (torch.int32, torch.int64)
                    or (d.action_indices < 0).any()):
                raise ValueError("expected nonnegative integer latent-action indices")
            future = d.future_observations
            if (future is None or isinstance(future, (list, tuple)) and not future
                    or isinstance(future, Tensor) and future.numel() == 0):
                raise ValueError("a privileged decision requires realized future observations")

@dataclass
class Query:
    observation: Any
    instruction: str
    hardware_id: str

class RobotBridge(Protocol):
    model: nn.Module

    def collect(self, task: dict) -> Sequence[Trajectory]:
        """Sample fresh closed-loop rollouts from model using current factors."""
        ...

    def action_logits(self, decision: Decision, *, privileged: bool) -> Tensor:
        """Return [tokens, action_vocabulary] with identical sampled prefixes.

        False: only information available at the original decision.
        True: same model/prefix plus realized future visual context.
        Never expose a sampled target token through the future attention path.
        """
        ...

    def reset(self, task: dict) -> Query: ...
    def encode_image(self, observation: Any) -> Tensor: ...
    def encode_text(self, instruction: str) -> Tensor: ...
    def act(self, observation: Any, instruction: str) -> Any:
        """Predict student-only actions and decode with the hardware action head."""
        ...
    def step(self, action: Any) -> tuple[Any, bool, dict]: ...

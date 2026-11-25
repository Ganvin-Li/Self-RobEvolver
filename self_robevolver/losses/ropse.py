"""ROPSE with stopped-gradient teacher targets and reliability weights."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.no_grad()
def future_progress(values: Tensor, gamma: float, horizon: int | None = None) -> Tensor:
    """max_{j>k} gamma**(j-k-1) [Q_j - Q_k]_+ for each step k."""
    if values.ndim != 1 or not values.is_floating_point():
        raise ValueError("quality scores must be a one-dimensional floating tensor")
    if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("quality scores must be finite and in [0, 1]")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if horizon is not None and (type(horizon) is not int or horizon < 1):
        raise ValueError("horizon must be positive")
    result = torch.zeros_like(values)
    for step in range(max(0, values.numel() - 1)):
        last = values.numel() if horizon is None else min(values.numel(), step + 1 + horizon)
        offsets = torch.arange(last - step - 1, device=values.device)
        gains = (values[step + 1 : last] - values[step]).clamp_min(0)
        result[step] = (gamma ** offsets * gains).max()
    return result


@torch.no_grad()
def teacher_confidence(teacher_action_logits: Tensor) -> Tensor:
    """Return detached confidence: 1 - H(p_T) / log(|Z|)."""
    if teacher_action_logits.ndim < 2 or teacher_action_logits.shape[-1] < 2:
        raise ValueError("logits must end in a latent action vocabulary of size >= 2")
    if not torch.isfinite(teacher_action_logits).all():
        raise ValueError("teacher logits must be finite")
    probs = teacher_action_logits.float().softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    return (1 - entropy / math.log(probs.shape[-1])).clamp(0, 1)


@torch.no_grad()
def gate_weights(
    progress: Tensor,
    confidence: Tensor,
    *,
    min_progress: float = 0.0,
    min_confidence: float = 0.0,
) -> Tensor:
    """Return sg[q * c * I(q >= delta_q) * I(c >= delta_c)]."""
    if progress.shape != confidence.shape:
        raise ValueError("progress and confidence must have identical shapes")
    if not 0 <= min_progress <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("invalid gate thresholds")
    if (not torch.isfinite(progress).all() or not torch.isfinite(confidence).all()
            or ((progress < 0) | (progress > 1)).any()
            or ((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("progress and confidence must be finite and in [0, 1]")
    valid = (progress >= min_progress) & (confidence >= min_confidence)
    return torch.where(valid, progress.clamp_min(0) * confidence.clamp(0, 1), torch.zeros_like(progress))


def weighted_forward_kl(
    student_action_logits: Tensor,
    teacher_action_logits: Tensor,
    weights: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """sum(sg[w] * KL(sg[p_T] || p_S)) / (sum(sg[w]) + eps)."""
    numerator, denominator = weighted_forward_kl_terms(
        student_action_logits, teacher_action_logits, weights
    )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")
    return numerator / (denominator + eps)


def weighted_forward_kl_terms(
    student_action_logits: Tensor,
    teacher_action_logits: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the weighted KL sum and detached global normalization weight."""
    if student_action_logits.shape != teacher_action_logits.shape:
        raise ValueError("student and teacher action logits must match")
    if student_action_logits.shape[:-1] != weights.shape:
        raise ValueError("weights must match all non-vocabulary dimensions")
    if student_action_logits.ndim < 2 or student_action_logits.shape[-1] < 2:
        raise ValueError("invalid action vocabulary")
    if (not torch.isfinite(student_action_logits).all()
            or not torch.isfinite(teacher_action_logits).all()
            or not torch.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError("logits must be finite and weights finite and nonnegative")
    # Stop gradients through both teacher targets and weights in Eq. (6).
    teacher_logp = F.log_softmax(teacher_action_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_action_logits.float(), dim=-1)
    divergence = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True).sum(dim=-1)
    safe_weights = weights.detach().float()
    return (divergence * safe_weights).sum(), safe_weights.sum()

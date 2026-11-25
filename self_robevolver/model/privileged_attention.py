"""Shared-prefix student/teacher attention over realized future visual tokens."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor

@dataclass(frozen=True)
class PrivilegedLogits:
    student: Tensor
    teacher: Tensor
    prediction_positions: Tensor

def build_attention_masks(batch_size: int, prompt_length: int, action_length: int,
                          future_length: int, *, device: torch.device):
    """Return boolean allow masks [B,1,L,L] and next-token prediction positions."""
    lengths = (batch_size, prompt_length, action_length, future_length)
    if any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("all sequence lengths must be positive")
    total = prompt_length + action_length + future_length
    student = torch.ones(total, total, dtype=torch.bool, device=device).tril()
    action = slice(prompt_length, prompt_length + action_length)
    future = slice(prompt_length + action_length, total)
    # Appending future tokens preserves the deployed student's action positions.
    # Block action-to-future paths to prevent indirect target-token leakage.
    student[future, action] = False
    teacher = student.clone()
    positions = torch.arange(prompt_length - 1, prompt_length + action_length - 1,
                             device=device)
    teacher[positions[:, None], torch.arange(future.start, future.stop, device=device)] = True
    shape = (batch_size, -1, -1, -1)
    return student[None, None].expand(shape), teacher[None, None].expand(shape), positions

def additive_mask(allowed: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert allow bits to a decoder mask: allowed=0, blocked=-large."""
    result = torch.zeros(allowed.shape, device=allowed.device, dtype=dtype)
    return result.masked_fill(~allowed, torch.finfo(dtype).min)

def privileged_action_logits(
    decoder: Callable[[Tensor, Tensor], Tensor],
    embeddings: Tensor,
    prompt_length: int,
    action_length: int,
    future_length: int,
    action_token_ids: Tensor,
) -> PrivilegedLogits:
    """Use [prompt/current visual/history, sampled actions, future visuals].

    The shared decoder returns [batch, sequence, vocabulary], disables dropout,
    and honors the supplied 4D mask. Inputs are unpadded; future features are
    encoded by the frozen vision backbone from the same executed rollout.
    """
    if embeddings.ndim != 3 or embeddings.shape[1] != prompt_length + action_length + future_length:
        raise ValueError("embedding layout does not match the supplied lengths")
    if (action_token_ids.ndim != 1 or action_token_ids.numel() < 2
            or action_token_ids.dtype not in (torch.int32, torch.int64)
            or action_token_ids.unique().numel() != action_token_ids.numel()):
        raise ValueError("expected distinct IDs of the ordered action vocabulary")
    if isinstance(decoder, torch.nn.Module) and any(
            isinstance(module, torch.nn.modules.dropout._DropoutNd)
            and module.training and module.p > 0 for module in decoder.modules()):
        raise ValueError("disable decoder dropout for matched student/teacher forwards")
    student_mask, teacher_mask, positions = build_attention_masks(
        embeddings.shape[0], prompt_length, action_length, future_length,
        device=embeddings.device)
    student_full = decoder(embeddings, additive_mask(student_mask, embeddings.dtype))
    with torch.no_grad():
        teacher_full = decoder(embeddings.detach(), additive_mask(teacher_mask, embeddings.dtype)).detach()
    if (student_full.ndim != 3 or student_full.shape[:2] != embeddings.shape[:2]
            or teacher_full.shape != student_full.shape):
        raise ValueError("student and teacher decoders must return aligned full-vocabulary logits")
    ids = action_token_ids.to(device=student_full.device, dtype=torch.long)
    if ids.min() < 0 or ids.max() >= student_full.shape[-1]:
        raise ValueError("action token ID outside vocabulary")
    student = student_full[:, positions].index_select(-1, ids)
    teacher = teacher_full[:, positions].index_select(-1, ids)
    return PrivilegedLogits(student, teacher, positions)

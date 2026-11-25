"""Cosine decay with optional explicitly configured warmup."""
import math

def learning_rate(step: int, total_steps: int, peak: float = 1e-5,
                  minimum: float = 1e-6, warmup_ratio: float = 0.0) -> float:
    if (total_steps < 1 or not 0 <= step < total_steps
            or not 0 <= minimum <= peak or peak <= 0 or not 0 <= warmup_ratio < 1):
        raise ValueError("invalid schedule")
    if total_steps == 1:
        return peak
    # Leave at least two cosine points (peak and minimum) when possible.
    warmup = min(int(total_steps * warmup_ratio), max(0, total_steps - 2))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))

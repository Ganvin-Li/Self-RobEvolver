"""Algorithm 1: on-policy self-evolution with structured lifelong memory."""
from __future__ import annotations
from collections.abc import Iterable
import torch
from torch import Tensor
from .interfaces import RobotBridge
from .task_registry import FactorRoute, TRAIN_TASK_IDS
from .model.adapters import begin_task_all, adapter_parameters, adapter_state
from .model.tucker6d import PARAMETER_NAMES, protected_optimizer_step, use_factor_route
from .losses.ropse import future_progress, teacher_confidence, gate_weights, weighted_forward_kl_terms
from .memory.fisher import FisherMemory
from .memory.retrieval import FactorPrototypeMemory
from .schedule import learning_rate


def validate_run_config(config: dict) -> None:
    """Require explicit values for settings not specified numerically in the paper."""
    required = {"optimization": ("updates_per_task", "weight_decay"),
                "ropse": ("gamma", "min_progress", "min_confidence")}
    missing = [f"{section}.{name}" for section, names in required.items()
               for name in names if config.get(section, {}).get(name) is None]
    if missing:
        raise ValueError("provide run-specific settings: " + ", ".join(missing))
    opt, r = config["optimization"], config["ropse"]
    if type(opt["updates_per_task"]) is not int or opt["updates_per_task"] < 1:
        raise ValueError("updates_per_task must be a positive integer")
    if not 0 <= float(opt["weight_decay"]) < float("inf"):
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 < float(r["gamma"]) <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if any(not 0 <= float(r[key]) <= 1 for key in ("min_progress", "min_confidence")):
        raise ValueError("ROPSE thresholds must be in [0, 1]")

def estimate_fisher(adapters, token_log_probabilities: Iterable[Tensor]):
    """Sum per-token squared score gradients from fresh final-student samples."""
    entries = [(path, name, getattr(module, name))
               for path, module in adapters.items() for name in PARAMETER_NAMES]
    sums = {path: {name: torch.zeros_like(getattr(module, name), device="cpu")
                   for name in PARAMETER_NAMES} for path, module in adapters.items()}
    count = 0
    for logp in token_log_probabilities:
        gradients = torch.autograd.grad(logp, [p for _, _, p in entries],
                                        allow_unused=True, retain_graph=True)
        for (path, name, _), gradient in zip(entries, gradients):
            if gradient is not None:
                sums[path][name] += gradient.detach().float().cpu().square()
        count += 1
    if not count:
        raise ValueError("Fisher needs fresh student samples")
    return sums, count

class LifelongLearner:
    def __init__(self, bridge: RobotBridge, adapters: dict, config: dict):
        if not adapters:
            raise ValueError("no installed adapters")
        validate_run_config(config)
        self.bridge, self.adapters, self.config = bridge, adapters, config
        self.fisher = FisherMemory()
        self.prototypes = FactorPrototypeMemory()
        self.trained_task_ids: list[int] = []

    def train_task(self, task: dict):
        task_id = task.get("task_id")
        if type(task_id) is not int or task_id not in TRAIN_TASK_IDS:
            raise ValueError("only tasks 1--30 may update the model or memories")
        if task_id != len(self.trained_task_ids) + 1:
            raise ValueError("lifelong tasks must be learned once in order, from 1 to 30")
        route = FactorRoute(**task["route"])
        new = begin_task_all(self.adapters, route)
        new_axes = next(iter(new.values()))
        if any(axes != new_axes for axes in new.values()):
            raise ValueError("inconsistent factor banks")
        cfg = self.config
        opt, reg, r = cfg["optimization"], cfg["regularization"], cfg["ropse"]
        optimizer = torch.optim.AdamW(adapter_parameters(self.adapters),
                                      lr=float(opt["learning_rate"]),
                                      weight_decay=float(opt["weight_decay"]))
        self.prototypes.register_hardware(task["hardware_id"], route.embodiment)
        # Disable dropout without disabling student gradients.
        self.bridge.model.eval()
        steps = int(opt["updates_per_task"])
        for update in range(steps):
            lr = learning_rate(update, steps, float(opt["learning_rate"]),
                               float(opt["min_learning_rate"]), float(opt.get("warmup_ratio", 0.0)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            with use_factor_route(route), torch.no_grad():
                trajectories = self.bridge.collect(task)
            if not trajectories:
                raise ValueError("collect must return actual on-policy trajectories")
            numerator = next(iter(self.adapters.values())).core.new_zeros(())
            denominator = numerator.detach().clone()
            for trajectory in trajectories:
                trajectory.validate()
                quality = future_progress(trajectory.quality.detach(), float(r["gamma"]))
                for k, decision in enumerate(trajectory.decisions):
                    self.bridge.model.eval()
                    with use_factor_route(route):
                        student = self.bridge.action_logits(decision, privileged=False)
                    if student.ndim != 2 or student.shape[0] != decision.action_indices.numel():
                        raise ValueError("action logits must align with the sampled action prefix")
                    with use_factor_route(route), torch.no_grad():
                        teacher = self.bridge.action_logits(decision, privileged=True)
                    confidence = teacher_confidence(teacher)
                    q = quality[k].to(confidence.device).expand_as(confidence)
                    weights = gate_weights(q, confidence,
                                           min_progress=float(r["min_progress"]),
                                           min_confidence=float(r["min_confidence"]))
                    num, den = weighted_forward_kl_terms(student, teacher, weights)
                    numerator, denominator = numerator + num, denominator + den
            # Normalize once across all rollouts; the denominator is not trainable.
            distillation = numerator / (denominator.detach() + 1e-8)
            inheritance = sum(m.regularization(lambda_orth=float(reg["lambda_orth"]),
                                               lambda_cons=float(reg["lambda_cons"]))
                              for m in self.adapters.values())
            retention = (numerator.new_zeros(()) if new_axes else
                         float(reg["lambda_fisher"]) * self.fisher.penalty(self.adapters))
            loss = distillation + inheritance + retention
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            protected_optimizer_step(tuple(self.adapters.values()), optimizer)
            yield {"task": task["name"], "update": update + 1, "learning_rate": lr,
                   "ropse": float(distillation.detach()), "weight_sum": float(denominator),
                   "loss": float(loss.detach())}

        # Fresh final-policy rollouts for Fisher; do not reuse pre-update actions.
        self.bridge.model.eval()
        with use_factor_route(route), torch.no_grad():
            final_rollouts = self.bridge.collect(task)
        if not final_rollouts:
            raise ValueError("final-policy rollouts are required")

        def log_probabilities():
            for trajectory in final_rollouts:
                trajectory.validate()
                for decision in trajectory.decisions:
                    self.bridge.model.eval()
                    with use_factor_route(route):
                        logits = self.bridge.action_logits(decision, privileged=False)
                    ids = decision.action_indices.to(device=logits.device, dtype=torch.long)
                    if logits.ndim != 2 or logits.shape[0] != ids.numel():
                        raise ValueError("action logits and sampled tokens are misaligned")
                    if ((ids < 0) | (ids >= logits.shape[-1])).any():
                        raise ValueError("local action index outside the action vocabulary")
                    selected = logits.float().log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
                    yield from selected.unbind()

        gradients, count = estimate_fisher(self.adapters, log_probabilities())
        self.fisher.consolidate(self.adapters, gradients, count,
                                smoothing=float(reg["fisher_smoothing"]))
        for trajectory in final_rollouts:
            self.prototypes.add_rollout(route, trajectory.image_features, trajectory.text_features)
        self.trained_task_ids.append(task_id)

    def state_dict(self):
        return {"version": 1, "adapters": adapter_state(self.adapters),
                "fisher": self.fisher.state_dict(), "prototypes": self.prototypes.state_dict(),
                "trained_task_ids": list(self.trained_task_ids),
                "config": self.config}

"""Independent factor search from frozen training-rollout CLIP features."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
from ..task_registry import FactorRoute

AXES = ("scene", "environment", "instruction_style")

def unit(vector: Tensor) -> Tensor:
    value = vector.detach().float().cpu()
    if value.ndim != 1 or not torch.isfinite(value).all() or value.norm() <= 0:
        raise ValueError("expected one finite nonzero feature vector")
    return F.normalize(value, dim=0)

class FactorPrototypeMemory:
    def __init__(self):
        self.sums: dict[str, dict[str, Tensor]] = {a: {} for a in AXES}
        self.counts: dict[str, dict[str, int]] = {a: {} for a in AXES}
        self.hardware: dict[str, str] = {}

    def register_hardware(self, identity: str, embodiment: str):
        if not identity or not embodiment:
            raise ValueError("empty hardware identity or embodiment")
        if identity in self.hardware and self.hardware[identity] != embodiment:
            raise ValueError("hardware mapping changed")
        self.hardware[identity] = embodiment

    def add(self, axis: str, category: str, feature: Tensor):
        if axis not in AXES or not category:
            raise ValueError("unknown axis or empty category")
        vector = unit(feature)
        old = self.sums[axis].get(category)
        if old is not None and old.shape != vector.shape:
            raise ValueError("feature dimensions changed")
        self.sums[axis][category] = vector.clone() if old is None else old + vector
        self.counts[axis][category] = self.counts[axis].get(category, 0) + 1

    def add_rollout(self, route: FactorRoute, image_features: Tensor, text_features: Tensor):
        if (image_features.ndim != 2 or text_features.ndim != 2
                or image_features.shape[0] != text_features.shape[0] or not len(image_features)):
            raise ValueError("expected aligned nonempty [steps, feature] matrices")
        for visual, text in zip(image_features, text_features):
            self.add("scene", route.scene, visual)
            self.add("environment", route.environment, visual)
            self.add("instruction_style", route.instruction_style, text)

    def retrieve(self, image_feature: Tensor, text_feature: Tensor, hardware_id: str):
        embodiment = self.hardware[hardware_id]
        selected, scores = {}, {}
        for axis in AXES:
            query = unit(text_feature if axis == "instruction_style" else image_feature)
            if not self.sums[axis]:
                raise ValueError(f"no learned {axis} prototypes")
            scores[axis] = {name: float(query @ unit(total))
                            for name, total in sorted(self.sums[axis].items())}
            selected[axis] = max(scores[axis], key=scores[axis].get)
        route = FactorRoute(embodiment, selected["scene"], selected["environment"],
                            selected["instruction_style"])
        return route, scores

    def state_dict(self):
        return {"version": 1, "sums": self.sums, "counts": self.counts,
                "hardware": self.hardware}

    @classmethod
    def from_state_dict(cls, state):
        if state["version"] != 1:
            raise ValueError("unsupported prototype memory")
        obj = cls()
        obj.sums = {a: dict(state["sums"][a]) for a in AXES}
        obj.counts = {a: dict(state["counts"][a]) for a in AXES}
        obj.hardware = dict(state["hardware"])
        return obj

"""LSERL factor routes and the 30-learning/10-held-out task protocol."""
from dataclasses import dataclass

TRAIN_TASK_IDS = tuple(range(1, 31))
HELD_OUT_TASK_IDS = tuple(range(31, 41))

@dataclass(frozen=True)
class FactorRoute:
    embodiment: str
    scene: str
    environment: str
    instruction_style: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in self.categories()):
            raise ValueError("factor categories must be nonempty strings")

    def categories(self) -> tuple[str, str, str, str]:
        return self.embodiment, self.scene, self.environment, self.instruction_style


def validate_task_sequence(sequence: dict) -> tuple[list[dict], list[dict]]:
    """Require 30 sequential learning tasks and 10 unseen factor combinations."""
    if not isinstance(sequence, dict):
        raise ValueError("expected an LSERL task sequence")
    training, held_out = sequence.get("training"), sequence.get("evaluation")
    if (not isinstance(training, list) or not isinstance(held_out, list)
            or len(training) != len(TRAIN_TASK_IDS) or len(held_out) != len(HELD_OUT_TASK_IDS)):
        raise ValueError("supply the actual 40-task schedule: 30 training tasks and 10 held-out tasks")
    names, hardware = set(), {}
    routes = []
    for tasks, expected in ((training, TRAIN_TASK_IDS), (held_out, HELD_OUT_TASK_IDS)):
        if any(not isinstance(task, dict) for task in tasks):
            raise ValueError("each task must be a mapping")
        ids = [task.get("task_id") for task in tasks]
        if any(type(task_id) is not int for task_id in ids) or tuple(ids) != expected:
            raise ValueError("training IDs must be 1--30 in order; held-out IDs must be 31--40")
        for task in tasks:
            name, identity = task.get("name"), task.get("hardware_id")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("task names must be unique nonempty strings")
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("each task requires a hardware identity")
            names.add(name)
            route = FactorRoute(**task["route"])
            if hardware.setdefault(identity, route.embodiment) != route.embodiment:
                raise ValueError("a hardware identity cannot change embodiment")
            routes.append(route.categories())
    learned = routes[:len(TRAIN_TASK_IDS)]
    known = [set(values) for values in zip(*learned)]
    for route in routes[len(TRAIN_TASK_IDS):]:
        if route in learned or any(value not in bank for value, bank in zip(route, known)):
            raise ValueError("held-out tasks must be new combinations of learned factor categories")
    return training, held_out

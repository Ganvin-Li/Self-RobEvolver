"""Algorithm 2: retrieve factor memories once, then act without future context."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from .train import load_factory
from .model.adapters import restore_tucker_adapters
from .model.tucker6d import use_factor_route
from .memory.retrieval import FactorPrototypeMemory
from .task_registry import TRAIN_TASK_IDS, validate_task_sequence

@torch.inference_mode()
def infer_episode(bridge, prototypes, task, max_steps):
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    bridge.model.eval()
    query = bridge.reset(task)
    route, scores = prototypes.retrieve(bridge.encode_image(query.observation),
                                        bridge.encode_text(query.instruction), query.hardware_id)
    observation, done, info = query.observation, False, {}
    with use_factor_route(route):
        for step in range(max_steps):
            action = bridge.act(observation, query.instruction)
            observation, done, info = bridge.step(action)
            if done:
                break
    return {"task": task["name"], "retrieved_factors": route.__dict__,
            "retrieval_scores": scores, "steps": step + 1, "terminated": done,
            "metrics": info}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default="config/task.yaml")
    parser.add_argument("--bridge", required=True, help="your_module:build_bridge")
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    _, tasks = validate_task_sequence(yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8")))
    # Load only trusted locally produced checkpoints.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(state.get("trained_task_ids", ())) != TRAIN_TASK_IDS:
        raise ValueError("held-out evaluation requires a checkpoint learned from tasks 1--30 only")
    bridge = load_factory(args.bridge)(state["config"])
    restore_tucker_adapters(bridge.model, state["adapters"])
    prototypes = FactorPrototypeMemory.from_state_dict(state["prototypes"])
    for task in tasks:
        print(json.dumps(infer_episode(bridge, prototypes, task, args.max_steps)))

if __name__ == "__main__":
    main()

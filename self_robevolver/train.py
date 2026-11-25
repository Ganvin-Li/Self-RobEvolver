"""Train through an explicitly supplied backbone/environment bridge."""
import argparse
import importlib
import json
from pathlib import Path
import torch
import yaml
from .model.adapters import install_tucker_adapters
from .learner import LifelongLearner, validate_run_config
from .task_registry import validate_task_sequence

def load_factory(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("--tasks", default="config/task.yaml")
    parser.add_argument("--bridge", required=True, help="your_module:build_bridge; no simulator is bundled")
    parser.add_argument("--output", default="outputs/self_robevolver")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tasks, _ = validate_task_sequence(yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8")))
    validate_run_config(cfg)
    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg["seed"]))
    bridge = load_factory(args.bridge)(cfg)
    adapters = install_tucker_adapters(bridge.model, **cfg["adapter"])
    learner = LifelongLearner(bridge, adapters, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("task_*.pt")):
        raise FileExistsError("use a new output directory; automatic resume is outside this core release")
    for index, task in enumerate(tasks, 1):
        for report in learner.train_task(task):
            print(json.dumps(report), flush=True)
        torch.save(learner.state_dict(), output / f"task_{index:03d}.pt")

if __name__ == "__main__":
    main()

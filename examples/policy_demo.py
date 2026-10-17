"""Run the saved actor on an explicitly synthetic observation fixture."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tennis_control.policy import load_policy, predict


def run():
    root = Path(__file__).resolve().parents[1]
    observations = json.loads((root / "examples" / "observations.json").read_text())
    actions = predict(load_policy(root / "run-seed-0" / "checkpoint_actor.pth"), observations)
    return {
        "scope": "CPU inference on synthetic inputs; not a Unity rollout or evaluation",
        "observations_shape": [len(observations), 24],
        "actions_shape": [len(actions), 2],
        "actions": [[round(value, 6) for value in row] for row in actions],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

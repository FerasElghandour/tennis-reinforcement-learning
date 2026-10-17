"""Exercise the shared actor/critic update using generated transitions on CPU.

This is an optimization example, not Tennis environment training.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tennis_control import maddpg_agent as implementation


def run():
    torch.set_num_threads(1)
    implementation.device = torch.device("cpu")
    agent = implementation.MADDPGAgent(24, 2, 2, 0)
    before = [p.detach().clone() for p in agent.actor_local.parameters()]
    rng = np.random.default_rng(42)
    for _ in range(implementation.BATCH_SIZE // 2):
        states = rng.normal(size=(2, 24)).astype(np.float32)
        actions = agent.act(states, add_noise=False)
        agent.step(states, actions, [.1, 0.0], states * .95, [False, False])
    changed = any(not torch.equal(a, b) for a, b in zip(before, agent.actor_local.parameters()))
    assert changed and agent.last_losses
    assert all(np.isfinite(value) for value in agent.last_losses.values())
    return {
        "scope": "synthetic transition training example; not a Unity rollout",
        "device": "cpu", "replay_transitions": len(agent.memory),
        "minibatch_size": implementation.BATCH_SIZE,
        "actor_weights_changed": changed,
        "last_losses": {k: round(v, 6) for k, v in agent.last_losses.items()},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

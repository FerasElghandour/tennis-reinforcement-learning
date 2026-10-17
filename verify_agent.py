"""Smoke test for a learning update and a saved/reloaded policy."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from maddpg_agent import MADDPGAgent, BATCH_SIZE
from model import Actor

torch.set_num_threads(1)
a = MADDPGAgent(24,2,2,0)
for _ in range(BATCH_SIZE//2):
    states=np.random.normal(size=(2,24)).astype(np.float32)
    a.step(states, a.act(states), [0.1,0.0], states*0.95, [False,False])
assert all(p.grad is None for p in a.actor_target.parameters())
assert all(p.grad is None for p in a.critic_target.parameters())
assert a.actor_local.bn1.num_batches_tracked.item()>0
assert all(torch.isfinite(p).all() for p in a.actor_local.parameters())
actions=a.act(np.ones((2,24),dtype=np.float32),add_noise=False)
assert actions.shape==(2,2) and np.isfinite(actions).all()
assert np.max(np.abs(actions))<=1
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'actor.pth'
    torch.save(a.actor_local.state_dict(),p)
    restored=Actor(24,2,0)
    restored.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
    restored.eval()
    with torch.no_grad(): np.testing.assert_allclose(restored(torch.ones(2,24)).numpy(), actions)
print('PASS: learning update, finite bounded actions, frozen target gradients, saved/reloaded policy')

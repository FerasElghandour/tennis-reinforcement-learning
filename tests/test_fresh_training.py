from types import SimpleNamespace
import numpy as np
import torch
from unittest.mock import patch
from fresh_training import rollout, evaluate

class FakeEnvironment:
    brain_names=['tennis']
    def __init__(self):self.closed=False
    def reset(self,train_mode):return {'tennis':SimpleNamespace(vector_observations=np.zeros((2,24)),agents=[0,1])}
    def step(self,actions):
        assert actions.shape==(2,2)
        return {'tennis':SimpleNamespace(vector_observations=np.ones((2,24)),agents=[0,1],rewards=[.1,-.01],local_done=[True,True])}
    def close(self):self.closed=True

def test_episode_uses_max_undiscounted_agent_return():
    r=rollout(FakeEnvironment(),'tennis')
    assert r['episode_score']==.1 and r['agent_1_return']==-.01
    assert r['steps']==1 and not r['truncated']

def test_evaluation_restores_training_rng_and_closes_environment():
    env=FakeEnvironment();before=torch.get_rng_state().clone();np_before=np.random.get_state()
    def create(*args):torch.rand(2);np.random.rand(2);return env
    with patch('fresh_training.environment',side_effect=create):rows=evaluate('fixture',7,1,2,10,policy='random')
    assert len(rows)==2 and env.closed
    assert torch.equal(before,torch.get_rng_state())
    np.testing.assert_equal(np_before[1],np.random.get_state()[1])

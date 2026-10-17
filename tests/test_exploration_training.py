import numpy as np
from exploration_training import WarmupAgent

def test_warmup_is_seeded_and_evaluation_does_not_consume_its_budget():
    agent=WarmupAgent(24,2,2,5,warmup=2,sigma=.3);states=np.zeros((2,24),dtype=np.float32)
    expected=np.random.default_rng(17005)
    agent.act(states,False);assert agent.behavior_steps==0
    for _ in range(2):np.testing.assert_array_equal(agent.act(states,True),expected.uniform(-1,1,(2,2)).astype(np.float32))
    agent.noise.sample=lambda:np.zeros((2,2))
    np.testing.assert_array_equal(agent.act(states,True),agent.act(states,False))
    assert agent.behavior_steps==3 and agent.noise.sigma==.3

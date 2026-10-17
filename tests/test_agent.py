import numpy as np
import pytest

torch=pytest.importorskip('torch')
from maddpg_agent import MADDPGAgent, OUNoise, ReplayBuffer, BATCH_SIZE


def test_noise_uses_local_rng_and_reproduces_without_global_interference():
    a,b=OUNoise((2,2),7),OUNoise((2,2),7)
    for _ in range(3):
        np.random.seed(999);np.random.normal(size=300)
        np.testing.assert_array_equal(a.sample(),b.sample())


def test_replay_owns_its_data_and_is_not_reseeded_by_other_buffers():
    a,b=ReplayBuffer(2,100,4,7),ReplayBuffer(2,100,4,7)
    for i in range(10):
        state=np.full(3,i,dtype=np.float32)
        for buffer in [a,b]:buffer.add(state,[0,0],i,state,False)
        state[:]=-999
    ReplayBuffer(2,100,4,999)
    np.testing.assert_array_equal(a.sample()[0].cpu(),b.sample()[0].cpu())
    assert not any((e.state==-999).any() for e in a.memory)


def test_invalid_agent_shapes_and_values_fail_before_replay_mutation():
    a=MADDPGAgent(3,2,2,0)
    with pytest.raises(ValueError,match='States'):a.act(np.zeros((1,3)))
    with pytest.raises(ValueError,match='Rewards'):a.step(np.zeros((2,3)),np.zeros((2,2)),[1],np.zeros((2,3)),[False,False])
    assert len(a.memory)==0


def test_inference_preserves_eval_mode_and_has_no_parameter_gradients():
    a=MADDPGAgent(3,2,2,0);a.actor_local.eval()
    actions=a.act(np.ones((2,3)),add_noise=False)
    assert not a.actor_local.training
    assert actions.shape==(2,2) and np.abs(actions).max()<=1
    assert all(p.grad is None for p in a.actor_local.parameters())


def test_learning_update_records_finite_losses_and_freezes_target_gradients():
    torch.set_num_threads(1)
    a=MADDPGAgent(3,2,2,0);rng=np.random.default_rng(42)
    for _ in range(BATCH_SIZE//2):
        states=rng.normal(size=(2,3)).astype(np.float32)
        a.step(states,np.zeros((2,2)),[.1,.1],states*.9,[False,False])
    assert a.last_losses and all(np.isfinite(v) for v in a.last_losses.values())
    assert all(p.grad is None for p in a.actor_target.parameters())
    assert all(p.grad is None for p in a.critic_target.parameters())

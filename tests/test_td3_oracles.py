"""Numerical behavior checks protecting all three defining TD3 mechanisms."""
import numpy as np
import torch
from tennis_control.td3 import TD3Agent, bellman_target, smooth_actions

def test_clipped_twin_target_terminal_mask():
    rewards=torch.tensor([[1.],[2.],[3.]])
    dones=torch.tensor([[0.],[1.],[0.]])
    q1=torch.tensor([[10.],[100.],[7.]])
    q2=torch.tensor([[4.],[90.],[12.]])
    torch.testing.assert_close(bellman_target(rewards,dones,q1,q2,.9),torch.tensor([[4.6],[2.],[9.3]]))

def test_target_smoothing_has_two_distinct_clips():
    actions=torch.tensor([[.9,-.9],[.1,-.1]])
    noise=torch.tensor([[4.,-4.],[.2,-.2]])
    torch.testing.assert_close(smooth_actions(actions,noise,.5),torch.tensor([[1.,-1.],[.3,-.3]]))

def test_delayed_actor_and_target_updates_and_gradients():
    torch.manual_seed(7);agent=TD3Agent(seed=7)
    batch=(torch.randn(128,24),torch.rand(128,2)*2-1,torch.randn(128,1),torch.randn(128,24),torch.zeros(128,1))
    old_actor=[p.clone() for p in agent.actor_local.parameters()]
    old_q=[p.clone() for p in agent.q1.parameters()]
    old_target=[p.clone() for p in agent.actor_target.parameters()]
    agent.learn(batch)
    assert all(torch.equal(a,b) for a,b in zip(old_actor,agent.actor_local.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(old_q,agent.q1.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(old_target,agent.actor_target.parameters()))
    agent.learn(batch)
    assert any(not torch.equal(a,b) for a,b in zip(old_actor,agent.actor_local.parameters()))
    for old,local,target in zip(old_target,agent.actor_local.parameters(),agent.actor_target.parameters()):
        torch.testing.assert_close(target,.995*old+.005*local)
    assert agent.actor_updates==1
    assert all(p.grad is None for model in [agent.actor_target,agent.q1_target,agent.q2_target] for p in model.parameters())

def test_evaluation_does_not_consume_behavior_state():
    torch.manual_seed(9);a=TD3Agent(seed=9)
    state=np.zeros((2,24),dtype=np.float32)
    before=a.rng.bit_generator.state
    out=a.act(state,False)
    assert a.behavior_steps==0 and before==a.rng.bit_generator.state
    assert out.shape==(2,2) and np.isfinite(out).all() and (np.abs(out)<=1).all()

def test_learning_loss_uses_the_minimum_target_and_masks_terminals():
    class FixedCritic(torch.nn.Module):
        def __init__(self,bias,slope=0.):
            super().__init__();self.bias=torch.nn.Parameter(torch.tensor(float(bias)));self.slope=slope
        def forward(self,states,actions):return self.bias+self.slope*states[:,:1]
    torch.manual_seed(13);agent=TD3Agent(seed=13)
    agent.q1_target=FixedCritic(8.,1.);agent.q2_target=FixedCritic(2.,-1.)
    agent.q1=FixedCritic(0.);agent.q2=FixedCritic(1.)
    agent.critic_optimizer=torch.optim.SGD([*agent.q1.parameters(),*agent.q2.parameters()],lr=0.)
    states=torch.zeros(3,24);next_states=states.clone();next_states[:,0]=torch.tensor([1.,-4.,3.])
    agent.learn((states,torch.zeros(3,2),torch.tensor([[1.],[2.],[3.]]),next_states,torch.tensor([[0.],[1.],[0.]])))
    targets=torch.tensor([1.99,2.,2.01])
    torch.testing.assert_close(torch.tensor(agent.last_losses['critic1']),torch.mean(targets**2))
    torch.testing.assert_close(torch.tensor(agent.last_losses['critic2']),torch.mean((targets-1.)**2))
    assert all(p.grad is None for net in [agent.q1_target,agent.q2_target] for p in net.parameters())

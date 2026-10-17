"""Numerical checks of the DDPG learning target and target-state transfer."""
import numpy as np
import pytest
import torch
from tennis_control.maddpg_agent import MADDPGAgent

class TargetActor(torch.nn.Module):
    def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(.25))
    def forward(self,states):return self.value.expand(len(states),2)

class FormulaCritic(torch.nn.Module):
    def __init__(self,target=False):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(3. if target else 0.));self.target=target
    def forward(self,states,actions):
        return self.bias+actions[:,:1]*(2. if self.target else 1.)+(states[:,:1] if self.target else 0.)

def test_hand_computed_targets_terminal_mask_and_no_target_gradients():
    agent=MADDPGAgent(24,2,2,37)
    agent.actor_target=TargetActor();agent.critic_target=FormulaCritic(True)
    agent.critic_local=FormulaCritic();agent.critic_optimizer=torch.optim.SGD(agent.critic_local.parameters(),lr=0.)
    # Isolate this learning-target oracle; target-state updates have a separate test.
    agent._soft_update=lambda *args:None
    states=torch.zeros(3,24);next_states=states.clone();next_states[:,0]=torch.tensor([1.,2.,3.])
    experiences=(states,torch.zeros(3,2),torch.tensor([[1.],[2.],[3.]]),next_states,torch.tensor([[0.],[1.],[0.]]))
    agent._learn(experiences,.9)
    targets=np.array([5.05,2.,8.85])
    assert agent.last_losses['critic']==pytest.approx(float(np.mean(targets**2)),rel=1e-6)
    assert all(p.grad is None for p in agent.actor_target.parameters())
    assert all(p.grad is None for p in agent.critic_target.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in agent.actor_local.parameters())

def test_polyak_parameters_and_batchnorm_buffers_are_transferred():
    agent=MADDPGAgent(24,2,2,37)
    with torch.no_grad():
        for p in agent.actor_local.parameters():p.fill_(4.)
        for p in agent.actor_target.parameters():p.fill_(2.)
        agent.actor_local.bn1.running_mean.fill_(7.)
        agent.actor_local.bn1.running_var.fill_(9.)
        agent.actor_local.bn1.num_batches_tracked.fill_(11)
    agent._soft_update(agent.actor_local,agent.actor_target,.25)
    for p in agent.actor_target.parameters():torch.testing.assert_close(p,torch.full_like(p,2.5))
    for a,b in zip(agent.actor_local.buffers(),agent.actor_target.buffers()):torch.testing.assert_close(a,b)
    agent._hard_update(agent.actor_local,agent.actor_target)
    for key,value in agent.actor_local.state_dict().items():torch.testing.assert_close(value,agent.actor_target.state_dict()[key])

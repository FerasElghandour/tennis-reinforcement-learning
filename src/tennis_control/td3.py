"""Local-observation shared-policy TD3; independently implemented from published equations.

References: Fujimoto et al. (2018), https://arxiv.org/abs/1802.09477;
OpenAI Spinning Up, https://spinningup.openai.com/en/latest/algorithms/td3.html.
This is a shared policy, not a centralized multi-agent critic.
"""
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .maddpg_agent import ReplayBuffer


class Policy(nn.Module):
    def __init__(self, state_size=24, action_size=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, 256), nn.LayerNorm(256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size), nn.Tanh())
        nn.init.uniform_(self.net[-2].weight, -.003, .003)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, states):
        return self.net(states)


class Value(nn.Module):
    def __init__(self, state_size=24, action_size=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size + action_size, 256), nn.LayerNorm(256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


def smooth_actions(actions, noise, clip=.5):
    return (actions + noise.clamp(-clip, clip)).clamp(-1., 1.)


def bellman_target(rewards, dones, q1, q2, gamma=.99):
    return rewards + gamma * (1. - dones) * torch.minimum(q1, q2)


class TD3Agent:
    def __init__(self, state_size=24, action_size=2, num_agents=2, seed=0,
                 warmup=2000, sigma=.2):
        if torch.cuda.is_available():
            raise RuntimeError('This frozen reference implementation uses CPU replay and networks.')
        self.state_size, self.action_size, self.num_agents = state_size, action_size, num_agents
        self.seed, self.warmup, self.sigma = seed, warmup, sigma
        self.rng = np.random.default_rng(seed + 27000)
        self.target_rng = torch.Generator().manual_seed(seed + 37000)
        self.actor_local = Policy(state_size, action_size)
        self.actor_target = copy.deepcopy(self.actor_local)
        self.q1, self.q2 = Value(state_size, action_size), Value(state_size, action_size)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=.0003)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=.0003)
        self.memory = ReplayBuffer(action_size, 200000, 128, seed + 47000)
        self.noise = SimpleNamespace(state=np.zeros((num_agents, action_size)))
        self.behavior_steps = self.update_count = self.actor_updates = 0
        self.transitions = self.positive_transitions = self.clipped_actions = self.action_values = 0
        self.last_losses = None

    def reset(self):
        pass  # Gaussian noise is independent over time; no recurrent policy state.

    def act(self, states, add_noise=True):
        states = np.asarray(states, dtype=np.float32)
        if states.shape != (self.num_agents, self.state_size) or not np.isfinite(states).all():
            raise ValueError('Invalid observations')
        if add_noise:
            self.behavior_steps += 1
        if add_noise and self.behavior_steps <= self.warmup:
            actions = self.rng.uniform(-1., 1., (self.num_agents, self.action_size)).astype(np.float32)
        else:
            with torch.no_grad():
                actions = self.actor_local(torch.from_numpy(states)).numpy()
            if add_noise:
                actions = actions + self.rng.normal(0., self.sigma, actions.shape)
        actions = np.clip(actions, -1., 1.).astype(np.float32)
        if add_noise:
            self.clipped_actions += int((np.abs(actions) >= .999).sum())
            self.action_values += actions.size
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s, a, r, ns, d)
        self.transitions += len(rewards)
        self.positive_transitions += int((np.asarray(rewards) > 0).sum())
        if self.behavior_steps >= self.warmup and len(self.memory) >= 128:
            self.learn(self.memory.sample())

    @torch.no_grad()
    def update_targets(self, tau=.005):
        for local, target in [(self.actor_local, self.actor_target), (self.q1, self.q1_target), (self.q2, self.q2_target)]:
            for p, tp in zip(local.parameters(), target.parameters()):
                tp.lerp_(p, tau)

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
        self.update_count += 1
        with torch.no_grad():
            noise = torch.randn((len(states), self.action_size), generator=self.target_rng) * .2
            target_actions = smooth_actions(self.actor_target(next_states), noise)
            target = bellman_target(rewards, dones, self.q1_target(next_states, target_actions), self.q2_target(next_states, target_actions))
        loss1 = nn.functional.mse_loss(self.q1(states, actions), target)
        loss2 = nn.functional.mse_loss(self.q2(states, actions), target)
        self.critic_optimizer.zero_grad()
        (loss1 + loss2).backward()
        self.critic_optimizer.step()
        self.last_losses = {'critic1': float(loss1.detach()), 'critic2': float(loss2.detach()), 'actor_updated': False}
        if self.update_count % 2 == 0:
            for p in self.q1.parameters():
                p.requires_grad_(False)
            actor_loss = -self.q1(states, self.actor_local(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for p in self.q1.parameters():
                p.requires_grad_(True)
            self.update_targets()
            self.actor_updates += 1
            self.last_losses.update(actor=float(actor_loss.detach()), actor_updated=True)

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import Actor, Critic

# ── Hyperparameters ────────────────────────────────────────────────────────────
BUFFER_SIZE   = int(1e6)  # replay buffer capacity
BATCH_SIZE    = 128       # minibatch size
GAMMA         = 0.99      # discount factor
TAU           = 1e-2      # soft-update interpolation factor
LR_ACTOR      = 1e-3      # actor learning rate
LR_CRITIC     = 1e-3      # critic learning rate
WEIGHT_DECAY  = 0         # L2 weight decay for critic
UPDATE_EVERY  = 2         # steps between each learning phase
NUM_UPDATES   = 2         # gradient updates per learning phase
GRAD_CLIP     = 1.0       # critic gradient clipping
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGAgent:
    """Multi-agent DDPG using self-play.

    Both agents share one Actor and one Critic network, and one replay buffer.
    Each agent's experience is treated as an independent (s, a, r, s') tuple.

    The critic evaluates each agent's local observation/action pair.
    """

    def __init__(self, state_size, action_size, num_agents, seed):
        self.state_size  = state_size
        self.action_size = action_size
        self.num_agents  = num_agents
        self.seed        = seed
        self.t_step      = 0
        self.last_losses = None
        if min(state_size, action_size, num_agents) < 1:
            raise ValueError('State size, action size and agent count must be positive.')

        # ── Shared Actor (local + target) ──────────────────────────────────
        self.actor_local  = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # ── Shared Critic (local + target) ─────────────────────────────────
        self.critic_local  = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Initialise target weights from local
        self._hard_update(self.actor_local,  self.actor_target)
        self._hard_update(self.critic_local, self.critic_target)

        # OU noise — one process per agent
        self.noise = OUNoise((num_agents, action_size), seed)

        # Shared replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # ── Interaction ────────────────────────────────────────────────────────────

    def step(self, states, actions, rewards, next_states, dones):
        """Save each agent's experience; learn on a fixed schedule."""
        states = self._states(states)
        next_states = self._states(next_states)
        actions = np.asarray(actions, dtype=np.float32)
        rewards, dones = np.asarray(rewards), np.asarray(dones)
        if actions.shape != (self.num_agents, self.action_size) or not np.isfinite(actions).all():
            raise ValueError('Actions must be finite and match the agent/action dimensions.')
        if np.any(np.abs(actions) > 1):
            raise ValueError('Actions must be bounded to [-1, 1].')
        if rewards.shape != (self.num_agents,) or not np.isfinite(rewards).all():
            raise ValueError('Rewards must contain one finite value per agent.')
        if dones.shape != (self.num_agents,) or not np.isin(dones, [False, True]).all():
            raise ValueError('Done flags must contain one boolean per agent.')
        # Each agent contributes independently to the shared buffer
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s, a, r, ns, d)

        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0 and len(self.memory) >= BATCH_SIZE:
            for _ in range(NUM_UPDATES):
                self._learn(self.memory.sample(), GAMMA)

    def act(self, states, add_noise=True):
        """Return actions for all agents using the shared Actor."""
        states = self._states(states)
        states = torch.from_numpy(states).float().to(device)

        was_training = self.actor_local.training
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().numpy()
        self.actor_local.train(was_training)

        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)

    def _states(self, states):
        states = np.asarray(states, dtype=np.float32)
        if states.shape != (self.num_agents, self.state_size) or not np.isfinite(states).all():
            raise ValueError('States must be finite and match the agent/observation dimensions.')
        return states

    def reset(self):
        self.noise.reset()

    # ── Learning ───────────────────────────────────────────────────────────────

    def _learn(self, experiences, gamma):
        """Update shared actor and critic from a sampled minibatch."""
        states, actions, rewards, next_states, dones = experiences

        # ── Critic update ─────────────────────────────────────────────────────
        self.actor_target.eval()
        self.critic_target.eval()
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected     = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIP)
        self.critic_optimizer.step()

        # ── Actor update ──────────────────────────────────────────────────────
        self.critic_local.eval()
        for param in self.critic_local.parameters():
            param.requires_grad_(False)
        actions_pred = self.actor_local(states)
        actor_loss   = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.last_losses = {'actor': float(actor_loss.detach().cpu()),
                            'critic': float(critic_loss.detach().cpu())}
        for param in self.critic_local.parameters():
            param.requires_grad_(True)
        self.critic_local.train()

        # ── Soft-update target networks ───────────────────────────────────────
        self._soft_update(self.critic_local, self.critic_target, TAU)
        self._soft_update(self.actor_local,  self.actor_target,  TAU)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _soft_update(self, local_model, target_model, tau):
        if not 0 <= tau <= 1:
            raise ValueError('Soft-update tau must be between zero and one.')
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)
        # BatchNorm statistics must travel with the trained weights.
        for tb, lb in zip(target_model.buffers(), local_model.buffers()):
            tb.copy_(lb)

    def _hard_update(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())


# ── Ornstein-Uhlenbeck Noise ───────────────────────────────────────────────────

class OUNoise:
    """Temporally correlated exploration noise."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.10):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size  = size
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.rng.standard_normal(self.size)
        self.state = x + dx
        return self.state


# ── Replay Buffer ──────────────────────────────────────────────────────────────

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-size replay buffer shared by all agents."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        if not 0 < batch_size <= buffer_size:
            raise ValueError('Batch size must be positive and no larger than buffer capacity.')
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(np.array(state, copy=True), np.array(action, copy=True),
                                      reward, np.array(next_state, copy=True), done))

    def sample(self):
        if len(self.memory) < self.batch_size:
            raise ValueError('Not enough replay transitions for a complete batch.')
        experiences = self.rng.sample(self.memory, k=self.batch_size)

        def to_tensor(arr):
            return torch.from_numpy(np.vstack(arr)).float().to(device)

        states      = to_tensor([e.state      for e in experiences if e])
        actions     = to_tensor([e.action     for e in experiences if e])
        rewards     = to_tensor([e.reward     for e in experiences if e])
        next_states = to_tensor([e.next_state for e in experiences if e])
        dones       = torch.from_numpy(
            np.vstack([e.done for e in experiences if e]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

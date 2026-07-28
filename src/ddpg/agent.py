import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg.networks import Actor, Critic
from ddpg.noise import OUNoise
from ddpg.replay_buffer import ReplayBuffer


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent (Lillicrap et al., 2015).

    Actor-critic, off-policy method for continuous action spaces. The actor
    outputs a deterministic action; the critic learns Q(s, a) and provides
    the gradient the actor climbs. Both networks have slowly-tracking target
    copies (soft update) to keep the critic's bootstrap target stable.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        initial_random_steps: int = 2_000,
        ou_theta: float = 0.15,
        ou_sigma: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.noise = OUNoise(action_dim, theta=ou_theta, sigma=ou_sigma)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, step: int, explore: bool = True) -> np.ndarray:
        """Pick an action for `state`.

        During the first `initial_random_steps`, actions are sampled uniformly
        at random regardless of the (still untrained) policy. This warm-up is
        what actually lets the agent stumble onto strong torques early on -
        without it, exploration noise added on top of a near-zero initial
        policy stays too small to ever discover a full swing-up.
        """
        if explore and step < self.initial_random_steps:
            return np.random.uniform(-self.max_action, self.max_action, size=self.actor.layer3.out_features)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_t).cpu().numpy().flatten()

        if explore:
            action = action + self.noise.sample()

        return np.clip(action, -self.max_action, self.max_action)

    def reset_noise(self):
        self.noise.reset()

    def store(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def ready_to_update(self) -> bool:
        return len(self.memory) >= self.batch_size

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device),
        )

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, local_net, target_net):
        for target_param, param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor"])
        self.critic_target.load_state_dict(checkpoint["critic"])

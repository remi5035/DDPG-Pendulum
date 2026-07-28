import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer of past transitions, sampled uniformly at random.

    Breaks the temporal correlation between consecutive environment steps so
    that minibatches used for gradient updates are closer to i.i.d.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.float32),
            torch.as_tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

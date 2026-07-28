import copy

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated exploration noise.

    Plain i.i.d. Gaussian noise averages out step to step and rarely pushes a
    torque-controlled system like the pendulum far enough to discover a full
    swing-up. The OU process is correlated in time (dx = theta*(mu-x) + sigma*dW),
    so a burst of exploration keeps pushing in the same direction for a few
    steps in a row, which is what DDPG's original paper uses for this exact
    reason.
    """

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

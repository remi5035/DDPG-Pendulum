import torch
import torch.nn as nn
import torch.nn.functional as F

# Small final-layer init keeps the very first actions/Q-values close to zero.
# With PyTorch's default init, the actor's pre-tanh output can be large enough
# to saturate tanh from step one, which flattens the gradient and stalls
# learning before it starts. This is the initialization used in the original
# DDPG paper (Lillicrap et al., 2015).
FINAL_LAYER_INIT = 3e-3


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        self.layer3.weight.data.uniform_(-FINAL_LAYER_INIT, FINAL_LAYER_INIT)
        self.layer3.bias.data.uniform_(-FINAL_LAYER_INIT, FINAL_LAYER_INIT)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.max_action * torch.tanh(self.layer3(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.layer3.weight.data.uniform_(-FINAL_LAYER_INIT, FINAL_LAYER_INIT)
        self.layer3.bias.data.uniform_(-FINAL_LAYER_INIT, FINAL_LAYER_INIT)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

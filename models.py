import torch
from torch import nn
import torch.nn.functional as F


class Transition(nn.Module):
    def __init__(
        self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 4
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.norm_1 = nn.BatchNorm(state_dim + action_dim)
        self.dense_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        # self.norm_2 = nn.BatchNorm(hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, state_dim)  # distr params

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, action], -1)
        # x = self.norm_1(x)
        x = F.leaky_relu(x)
        x = self.dense_1(x)
        # x = self.norm_2(x)
        x = F.leaky_relu(x)
        delta = self.dense_2(x)
        new_state = state + delta
        return new_state, delta

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    # see third bullet point under "2. Orthogonal Initialization of Weights and Constant Initialization of biases" in https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, torch.tensor(envs.single_action_space.shape).prod()), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, torch.tensor(envs.single_action_space.shape).prod())
        )

    def value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # During learning the agent explores the environment by sampling actions from a Normal distribution. The standard deviation is a learnable parameter that should decrease during training as the agent gets more confident in its actions.
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

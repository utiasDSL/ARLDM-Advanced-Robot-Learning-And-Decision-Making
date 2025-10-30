from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy  # noqa: F401
import torch
import torch.nn as nn
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import SimData
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array
from torch import Tensor
from torch.distributions import Normal


########################################################################
# TODO: We provide the original implementation of the FigureEightEnv.
# Feel free to customize anything necessary to train your policy. 
# In most cases, you will not need any additional Python libraries.
#
# NOTE:
# During evaluation, your policy will be tested on randomly generated 
# trajectories. The performance criterion is the Root Mean Squared Error (RMSE)
# of the tracking error.
#
# You may modify the way observations are constructed, but make sure that 
# your environment defines either the `self.trajectory` or `self.trajectories`
# attribute as the goal trajectory. These attributes will be overridden 
# during testing.
#
# Hints:
# 1. You can customize the trajectories used for training. Instead of the
#    fixed figure-eight trajectory, you may design diverse or random 
#    trajectories for each environment instance.
# 2. Check your reward curve in Weights & Biases (wandb) to ensure that 
#    your training process is converging.
########################################################################
class RandTrajEnv(DroneEnv):
    """Drone environment for following a random trajectory.
    
    This environment is used to follow a random trajectory. The observations contain the
    relative position errors to the next `n_samples` points that are distanced by `samples_dt`. The
    reward is based on the distance to the next trajectory point.
    """

    def __init__(
        self,
        n_samples: int = 10,
        samples_dt: float = 0.1,
        trajectory_time: float = 10.0,
        *,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["sys_id", "analytical"] | Physics = Physics.sys_id,
        freq: int = 50,
        device: str = "cpu",
    ):
        """Initialize the environment and create the figure-eight trajectory.

        Args:
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            num_envs: Number of environments to run in parallel.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            drone_model: Drone model of the environment.
            freq: Frequency of the simulation.
            device: Device to use for the simulation.
        """
        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            freq=freq,
            device=device,
        )
        if trajectory_time < self.max_episode_time:
            raise ValueError("Trajectory time must be greater than max episode time")

        ########################################################################
        # TODO: Your Implementation
        #
        # Hints:
        # 1. You can implement your own goal trajectories for training here.
        # 2. You may override `self.trajectory` or define `self.trajectories`
        #    to support different trajectories for each environment instance.
        # 3. You can customize the observations. If you do so, remember to update 
        #    the observation space accordingly. Observations are implemented as 
        #    a `Dict` object. You can add additional entries to this dictionary, 
        #    and the `FlattenJaxObservation` wrapper will automatically flatten 
        #    all entries into a single vector.
        # 4. Feel free to explore other approaches to improve the policy performance.
        ########################################################################
        
        # Save variables to self
        self.trajectory_time = trajectory_time
        self.n_samples = n_samples
        self.samples_dt = samples_dt
        # NOTE: You can ignore/remove these variables if you want to implement obs differently. 

        # Create the figure eight trajectory
        n_steps = int(np.ceil(trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi, n_steps)
        radius = 1  # Radius for the circles
        y = np.zeros_like(t)  # x is 0 everywhere
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        z = radius * np.sin(2 * t) + 2.0  # Scale amplitude for 1-meter diameter
        # NOTE: our test trajectories start from [0.0, 0.0, 2.0]
        self.trajectory = np.array([x, y, z]).T # (n_steps, 3)
        self.trajectories = None # should be (num_envs, n_steps, 3)

        # Define trajectory sampling parameters
        self.sample_offsets = np.array(np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int)

        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()} # original obs from drone_env
        spec["local_samples"] = spaces.Box(-np.inf, np.inf, shape=(3 * self.n_samples,)) # extra obs for tracking
        self.single_observation_space = spaces.Dict(spec) # update obs space
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds) # batch obs space for vector env

        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################

        if self.trajectories is None:
            # if trajectories are not implemented, make copies of the single trajectory
            self.trajectories = np.tile(self.trajectory[None, :, :], (num_envs, 1, 1)) # (num_envs, n_steps, 3)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        ########################################################################
        # TODO: Customize reset function here if necessary
        # Hints: 
        # 1. You can update your own random trajectories here.
        ########################################################################
        














        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        self._reset(options=options) # call jax rest function
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}

    def render(self):
        # only render the first world
        idx = np.clip(self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1)
        next_trajectory = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], idx]
        draw_line(self.sim, self.trajectories[0, 0:-1:2, :], rgba=np.array([1,1,1,0.4]), start_size=2.0, end_size=2.0)
        draw_line(self.sim, next_trajectory[0], rgba=np.array([1,0,0,1]), start_size=3.0, end_size=3.0)
        draw_points(self.sim, next_trajectory[0], rgba=np.array([1.0, 0, 0, 1]), size=0.01)
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        obs = super().obs()
        ########################################################################
        # TODO: Customize observations here if necessary.
        # The observations are implemented as a `Dict` object.
        # You can customize them by simply adding more entries to the dictionary. 
        # The `FlattenJaxObservation` wrapper will then automatically flatten 
        # all dictionary entries into a single observation vector.
        ########################################################################

        idx = np.clip(self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1)
        dpos = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], idx] - self.sim.data.states.pos
        obs["local_samples"] = dpos.reshape(-1, 3 * self.n_samples)

        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return obs

    def reward(self) -> Array:
        ########################################################################
        # TODO: Customize rewards here if necessary
        # NOTE: If you're not familiar with jax.numpy, feel free to use numpy instead.
        ########################################################################

        obs = self.obs()
        pos = obs["pos"] # (num_envs, 3)
        goal = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], self.steps][:, 0, :] # (num_envs, 3)

        # distance to next trajectory point
        norm_distance = jnp.linalg.norm(pos - goal, axis=-1)
        reward = jnp.exp(-2.0 * norm_distance) # encourage flying close to goal
        reward = jnp.where(self.terminated(), -1.0, reward) # penalize drones that crash into the ground

        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return reward

    @property
    def steps(self) -> Array:
        """The current step in the trajectory."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq) - 1

    @staticmethod
    def _reset_randomization(data: SimData, mask: Array) -> SimData:
        """Randomize the initial position and velocity of the drones.

        This function will get compiled into the reset function of the simulation. Therefore, it
        must take data and mask as input arguments and must return a SimData object.
        """
        # Sample initial position
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        # NOTE: try not to modify, our test trajectories start from [0.0, 0.0, 2.0]
        pmin, pmax = jnp.array([-0.1, -0.1, 1.9]), jnp.array([0.1, 0.1, 2.1])
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
        vel = jax.random.uniform(key=vel_key, shape=shape, minval=-0.1, maxval=0.1)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
        return data



def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    # see third bullet point under "2. Orthogonal Initialization of Weights and Constant Initialization of biases" in https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MyAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        ########################################################################
        # TODO: Customize the agent here if necessary.
        #
        # Hints:
        # 1. You can experiment with different hidden layer sizes. 
        #    Be careful about overfitting in reinforcement learning.
        # 2. The stochastic policy is implemented as an `actor_mean` network 
        #    along with `actor_logstd` as a learnable parameter. 
        #    You may tune the initial values of these parameters 
        #    to observe how they affect training stability and performance.
        ########################################################################
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
        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################

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

import logging  # noqa: I001
import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import crazyflow  # noqa: F401, register the gymnasium envs
import gymnasium
import gymnasium.spaces.utils
import gymnasium.wrappers.vector.jax_to_torch
import numpy as np
import torch
import torch.nn as nn
import wandb

from crazyflow.utils import enable_cache
from gymnasium.vector import VectorEnv
from ml_collections import ConfigDict
from torch.optim import AdamW
from exercise06.agent import Agent
from exercise06.wrappers import FlattenJaxObservation, AngleReward, RecordData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# jax
enable_cache()


def set_seeds(seed: int):
    ########################################################################
    # TODO:
    # Set the random seeds for the random, numpy, and torch libraries.
    # Do not forget to set the cudnn.deterministic and cudnn.benchmark
    # flags correctly!
    # For help visit this page:
    # https://pytorch.org/docs/stable/notes/randomness.html
    # Hints:
    # - This function does not return anything
    # - Set the seeds to the input argument "seed"
    ########################################################################
    








    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################
    pass


def make_envs(env_name: str, n_envs: int, n_eval_envs: int, train_device: str, **kwargs: dict[str, Any]) -> tuple[VectorEnv, VectorEnv]:
    """Creates vectorized training and evaluation environments for reinforcement learning.

    Args:
        env_name : str
            Name of environment.

        n_envs : int
            Number of parallel environments to use for training.

        n_eval_envs : int
            Number of parallel environments to use for evaluation.

        train_device : str
            Device used for training.
        
        **kwargs : dict
            Extra keyword arguments to pass to `gymnasium.make_vec()`.

    Returns:
        train_envs : VectorEnv
            The wrapped training environments.

        eval_envs : VectorEnv
            The wrapped evaluation environments.
    """
    # cpu is faster for smaller number of parallel envs (~64)
    env_device = "cuda" if n_envs > 64 and torch.cuda.is_available() else "cpu"
    train_envs = gymnasium.make_vec(
        env_name,
        num_envs=n_envs,
        freq=50,
        device=env_device,
        **kwargs,
    )
    eval_envs = gymnasium.make_vec(
        env_name,
        num_envs=n_eval_envs,
        freq=50,
        device=env_device,
        **kwargs,
    )

    ########################################################################
    # TODO:
    # 1. Wrap the train_envs in an AngleReward wrapper. You can check the code 
    # in wrappers.py. This is to penalize large roll, pitch, yaw angles.
    # 2. Wrap the train_envs in an FlattenJaxObservation wrapper. This is to 
    # transform dict type observations to flat array for the neural network.
    # 3. Wrap the train_envs in an JaxToTorch wrapper (as our environment is
    # implemented in Jax, but the DRL in PyTorch). You can use                                                                    
    # https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.JaxToTorch
    # Hint: set the device of the JaxToTorch wrapper to `train_device`.
    ########################################################################
    







    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    ########################################################################
    # TODO:
    # 1. Wrap the eval_envs in an AngleReward wrapper. You can check the code 
    # in wrappers.py. This is to penalize large roll, pitch, yaw angles.
    # 2. Wrap the eval_envs in an FlattenJaxObservation wrapper. This is to 
    # transform dict type observations to flat array for the neural network.
    # 3. Wrap the eval_envs in an JaxToTorch wrapper (as our environment is
    # implemented in Jax, but the DRL in PyTorch). You can use
    # and
    # https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.JaxToTorch
    # Hints:
    # - Set the device of the JaxToTorch wrapper to train_device.
    # NOTE: Normally, we would also wrap the eval_envs in a 
    # NormalizeObservation wrapper. 
    # https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeObservation
    # However, we don't need to normalize the observations for this environment. 
    # Think about:
    # - What is the purpose of the NormalizeObservation wrapper?
    # - Should the eval_envs update the normalization statistics of the
    #   NormalizeObservation wrapper?
    ########################################################################
    







    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################
    return train_envs, eval_envs

def save_model(
    agent: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_envs: gymnasium.vector.VectorEnv,
    path: str,
    save: bool = True,
) -> dict[str, Optional[object]]:
    """Saves the model, optimizer state, and observation normalization statistics to a file. This is necessary for checkpointing and later inference or resuming training.

    Args:
        agent : nn.Module
            The PyTorch model whose parameters should be saved.

        optimizer : torch.optim.Optimizer
            The optimizer used during training whose state will be saved.

        train_envs: gymnasium.vector.VectorEnv
            The environment for training this agent.
            
        path : strs
            The file path where the checkpoint dictionary should be saved.

        save : bool
            If True, saves the model to disk. If False, only returns the dictionary without saving.

    Returns:
        save_dict : dict[str, Optional[object]]
            A dictionary containing:
            - "model_state_dict": The model's parameters (state dict)
            - "optim_state_dict": The optimizer's state
            - "obs_mean": The mean of observations (for normalization)
            - "obs_var": The variance of observations (for normalization)
    """
    save_dict = {
        "model_state_dict": None,
        "optim_state_dict": None,
    }
    ########################################################################
    # TODO:
    # Populate the keys "model_state_dict", "optim_state_dict"
    # "obs_mean", and "obs_var" with the appropriate values.
    # You should use the 'state_dict()' function for retrieving the values
    # as specified here:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    # Hints:
    # Copy and move all tensors to CPU for saving to be sure the
    # parameters can be loaded on a CPU-only device.
    ########################################################################
    






    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    if hasattr(train_envs.unwrapped, "n_samples"):
        save_dict["n_samples"] = train_envs.unwrapped.n_samples
    if save:
        torch.save(save_dict, path)
    # the return value is only used for testing the function
    return save_dict


def evaluate_agent(
    envs: VectorEnv, agent: nn.Module, n_steps: int, device: str, seed: int | None = None
) -> tuple[list[float], list[int]]:
    eval_obs, _ = envs.reset(seed=seed)
    ep_rewards = torch.zeros(envs.num_envs, device=device)
    ep_steps = torch.zeros_like(ep_rewards)
    rewards, steps = [], []  # All eval rewards are at the same global step, so we average
    for _ in range(n_steps):
        with torch.no_grad():
            action, _, _, _ = agent.action_and_value(eval_obs, deterministic=True)
        eval_obs, reward, terminated, truncated, _ = envs.step(action)
        ep_rewards += reward
        ep_steps += 1
        done = terminated | truncated
        rewards.extend([r.item() for r in ep_rewards[done]])
        steps.extend([s.item() for s in ep_steps[done]])
        ep_rewards[done] = 0
        ep_steps[done] = 0
    return rewards, steps


class PPOTrainer:
    def __init__(self, config: ConfigDict, wandb_log: bool = False, agent_cls: type[nn.Module] = Agent, **kwargs: dict[str, Any]):
        self.config = config
        self.wandb_log = wandb_log

        if wandb_log:
            wandb.init(project=f"ARLDM-Exercise-PPO-{self.config.env_name}", config=None, reinit=True)
            self.config.update(wandb.config)
            if self.config.get("n_train_samples"):
                self.config.n_steps = self.config.n_train_samples // self.config.n_envs
            wandb.config.update(dict(self.config))

        set_seeds(self.config.seed)
        self.train_envs, self.eval_envs = make_envs(
            self.config.env_name, self.config.n_envs, self.config.n_eval_envs, self.config.device, **kwargs
        )

        self.config.batch_size = self.config.n_envs * self.config.n_steps  # 1024*16
        self.config.minibatch_size = self.config.batch_size // self.config.n_minibatches
        self.config.n_iterations = self.config.total_timesteps // self.config.batch_size
        self.config.total_timesteps = self.config.n_iterations * self.config.batch_size

        if self.config.n_iterations < 1:
            raise ValueError("Not enough iterations to train")

        self.agent = agent_cls(self.train_envs).to(self.config.device)
        self.optimizer = AdamW(self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5)

        self.rewards = torch.zeros(
            self.config.n_envs, dtype=torch.float32, device=self.config.device
        )
        self.autoreset = torch.zeros(self.config.n_envs, dtype=bool, device=self.config.device)

        # Initialize buffers
        self.obs_buffer = torch.zeros(
            (self.config.n_steps, self.config.n_envs)
            + self.train_envs.single_observation_space.shape
        ).to(self.config.device)
        self.actions_buffer = torch.zeros(
            (self.config.n_steps, self.config.n_envs) + self.train_envs.single_action_space.shape
        ).to(self.config.device)
        self.logprobs_buffer = torch.zeros((self.config.n_steps, self.config.n_envs)).to(
            self.config.device
        )
        self.rewards_buffer = torch.zeros((self.config.n_steps, self.config.n_envs)).to(
            self.config.device
        )
        self.dones_buffer = torch.zeros((config.n_steps, config.n_envs)).to(config.device)
        self.terminated_buffer = torch.zeros((self.config.n_steps, self.config.n_envs)).to(
            self.config.device
        )
        self.values_buffer = torch.zeros((self.config.n_steps, self.config.n_envs)).to(
            self.config.device
        )

        # Stats tracking setup
        self.global_step = 0
        self.train_rewards_hist = []
        self.train_rewards_steps = []
        self.eval_rewards_hist = []
        self.eval_rewards_steps = []
        self.last_eval = self.global_step

    def calculate_advantages(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the Generalized Advantage Estimation (GAE) for a batch of trajectories.

        This function computes the advantages and the estimated returns using GAE. It uses
        the rewards, values, and done flags stored in the agent's buffers during rollout.

        Args:
            next_value : torch.Tensor
                The estimated value for the state following the final step in the trajectory.
                Shape: [1, n_envs]

        Returns:
            returns : torch.Tensor
                The computed returns for each time step: returns = advantages + values.
                Shape: [n_steps, n_envs]

            advantages : torch.Tensor
                The advantage estimates computed using GAE.
                Shape: [n_steps, n_envs]
        """
        advantages = torch.zeros_like(self.rewards_buffer).to(
            self.config.device
        )  # (n_steps, n_envs)
        lastgaelam = 0
        for t in reversed(range(self.config.n_steps)):
            if t == self.config.n_steps - 1:
                nextnonterminal = 1.0 - self.terminated_buffer[t]  # (n_envs,)
                nextvalues = next_value  # (1, n_envs)
            else:
                nextnonterminal = 1.0 - self.terminated_buffer[t + 1]
                nextvalues = self.values_buffer[t + 1]

            ####################################################################
            # TODO:
            # Paper: https://arxiv.org/pdf/1506.02438
            # We already implemented the recursive advantage estimation below.
            # This corresponds to equation (16) in the paper.
            # The part that is missing is the TD residual (delta) that is
            # required for the advantage computation. You need to look at the
            # paper to implement the TD error as used in GAE. You can find the
            # required equation at the beginning of Section 3 or in Eq. (1).6
            # in the paper.
            # 1. Compute the TD residual.
            # Hints:
            #   - Do not forget to account for nextnonterminal!
            #   - Variables which are required:
            #       self.rewards_buffer[t]: the reward at timestep t
            #       self.values_buffer[t]: the estimated value at timestep t
            #       nextvalues: the estimated value at timestep t+1
            #       nextnonterminal: 0.0 if episode ended at t+1, else 1.0
            #   - Please use the variable`delta = ...`
            ########################################################################
            delta = torch.zeros_like(next_value)  # (1, n_envs)
            







            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################
            # calculate advantages using the TD error "delta"
            advantages[t] = lastgaelam = (
                delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            )

        returns = advantages + self.values_buffer

        return returns, advantages

    def calculate_approx_kl(self, ratio, logratio, clipfracs):
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
        return old_approx_kl, approx_kl, clipfracs

    def calculate_pg_loss(self, mb_advantages: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """Calculate the policy gradient loss using the clipped surrogate objective from PPO (Proximal Policy Optimization).

        Args:
            mb_advantages : torch.Tensor
                The advantage estimates for a minibatch. Shape: (minibatch_size,).

            ratio : torch.Tensor
                The probability ratio between the new policy and the old policy:
                ratio = pi_new / pi_old. Shape: (minibatch_size,).

        Returns:
            pg_loss : torch.Tensor
                The scalar policy loss (a single-element tensor) computed using the
                clipped surrogate objective from PPO.
        """
        ########################################################################
        # TODO:
        # 1. Compute the surrogate loss.
        # 2. Compute the clipped surrogate loss.
        # 3. Take the element-wise *max* of the two losses.
        # 4. Return the mean policy gradient loss over the minibatch.
        # Hints:
        #   - Refer to the loss function L(s,a,phi_k,phi)=... in
        #     https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations
        #   - PyTorch minimizes the loss, but we want to
        #     maximize the advantage. So when translating equations from the
        #     website,
        #     you need to use 'max' instead of 'min', and take the negative of
        #     the advantages!
        #   - Use `self.config.clip_coef` to clip the loss
        ########################################################################
        pg_loss = None
        







        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return pg_loss

    def calculate_v_loss(
        self,
        newvalue: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
        mb_inds: np.ndarray,
        if_clip: bool = False,
    ) -> torch.Tensor:
        """Calculate the value function loss for PPO (clipped and unclipped).

        This function computes the value loss either as the standard
        mean squared error (MSE) or using the PPO clipped value function loss,
        depending on `if_clip`.

        Args:
            newvalue : torch.Tensor
                Predicted values from the value function for the current minibatch.
                Shape: (minibatch_size, 1).

            b_returns : torch.Tensor
                The estimated returns (advantages + values) computed from GAE.
                Shape: (batch_size,).

            b_values : torch.Tensor
                The baseline predicted values from the value function before the update.
                Used for clipping reference.
                Shape: (batch_size,).

            mb_inds : np.ndarray
                Indices into the full batch for selecting the current minibatch.
                Shape: (minibatch_size,).

            if_clip : bool
                Whether to use the clipped value function loss (as in PPO) or standard MSE.

        Returns:
            v_loss : torch.Tensor
                The scalar value loss (a single-element tensor).
                (As usual, reduce the minibatch using the mean.)
        """
        newvalue = newvalue.view(-1)  # (n_envs,1) -> (n_envs)
        if if_clip:
            ####################################################################
            # TODO:
            # Compute the clipped value loss (v_loss).
            # Hint:
            #   - Use `self.config.clip_coef` to calculate the clipped loss.
            #   - Refer to the equations given in the notebook.
            ####################################################################
            









            ####################################################################
            #                           END OF YOUR CODE
            ####################################################################
            pass
        else:
            ####################################################################
            # TODO:
            # Compute the standard (unclipped) mean-squared error value loss
            # (v_loss) using the current value function prediction and actual
            # returns.
            ####################################################################
            



            ####################################################################
            #                           END OF YOUR CODE
            ####################################################################
            pass

        return v_loss

    def collect_samples(self, obs):
        steps = torch.zeros(self.config.n_envs, dtype=torch.int32, device=self.config.device)

        while any(active := (steps < self.config.n_steps)):
            with torch.no_grad():
                action, logprob, _, value = self.agent.action_and_value(obs)

            next_obs, reward, terminated, truncated, _ = self.train_envs.step(action)
            # self.train_envs.render()
            done = terminated | truncated
            self.rewards += reward
            self.rewards[self.autoreset] = 0
            self.train_rewards_hist.extend([r.item() for r in self.rewards[done]])
            self.train_rewards_steps.extend([self.global_step] * sum(done))
            if self.wandb_log and done.any():
                for r in self.rewards[done]:
                    wandb.log({"train/reward": r.item()}, step=self.global_step)

            # Add sample to buffer
            mask = active & ~self.autoreset
            self.obs_buffer[steps[mask], mask] = obs[mask]
            self.terminated_buffer[steps[mask], mask] = terminated[mask].float()
            self.dones_buffer[steps[mask], mask] = done[mask].float()
            self.values_buffer[steps[mask], mask] = value[mask].squeeze()
            self.actions_buffer[steps[mask], mask] = action[mask]
            self.logprobs_buffer[steps[mask], mask] = logprob[mask].squeeze()
            self.rewards_buffer[steps[mask], mask] = reward[mask]
            steps[mask] += 1
            self.global_step += mask.sum().item()

            obs = next_obs
            self.autoreset = done

        return obs

    def learn(self, obs):
        # Bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.value(obs).reshape(1, -1)  # (1, 1024)
            returns, advantages = self.calculate_advantages(
                next_value
            )  # (n_steps, n_envs) (n_steps, n_envs)

        # Flatten the batch
        b_obs = self.obs_buffer.reshape(
            (-1,) + self.train_envs.single_observation_space.shape
        )  # (n_steps * n_envs, n_ieter)
        b_logprobs = self.logprobs_buffer.reshape(-1)  # (n_steps * n_envs)
        b_actions = self.actions_buffer.reshape(
            (-1,) + self.train_envs.single_action_space.shape
        )  # (n_steps * n_envs, 4)
        b_advantages = advantages.reshape(-1)  # (n_steps * n_envs)
        b_returns = returns.reshape(-1)  # (n_steps * n_envs)
        b_values = self.values_buffer.reshape(-1)  # (n_steps * n_envs)

        # Optimizing the policy and value network
        b_inds = np.arange(self.config.batch_size)  # (n_envs * n_steps,)
        clipfracs = []
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl, approx_kl, clipfracs = self.calculate_approx_kl(
                        ratio, logratio, clipfracs
                    )

                mb_advantages = b_advantages[mb_inds]  # (n_envs)
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss = self.calculate_pg_loss(mb_advantages, ratio)

                # Value loss
                v_loss = self.calculate_v_loss(
                    newvalue, b_returns, b_values, mb_inds, if_clip=self.config.clip_vloss
                )

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.wandb_log:
            wandb.log(
                {
                    "train/value_loss": v_loss.item(),
                    "train/policy_loss": pg_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "train/old_approx_kl": old_approx_kl.item(),
                    "train/approx_kl": approx_kl.item(),
                    "train/clipfrac": np.mean(clipfracs),
                    "train/explained_var": explained_var,
                },
                step=self.global_step,
            )

    def train(self):
        # Warm-up steps to initialize environment (required for jax)
        obs, _ = self.train_envs.reset(seed=self.config.seed)
        _ = self.train_envs.action_space.seed(self.config.seed)
        for _ in range(100):
            self.train_envs.step(torch.tensor(self.train_envs.action_space.sample()))

        # Main training loop
        for iteration in range(1, self.config.n_iterations + 1):
            # Decay learning rate
            if self.config.lr_decay:
                progress = self.global_step / self.config.total_timesteps
                current_lr = self.config.learning_rate * (1 - 0.8 * progress)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr
                if self.wandb_log:
                    wandb.log({"train/learning_rate": current_lr}, step=self.global_step)

            start_time = time.time()

            # Collect samples and learn
            obs = self.collect_samples(obs)
            self.learn(obs)

            # Periodic evaluation
            if self.global_step - self.last_eval >= self.config.eval_interval:
                eval_rewards, eval_steps = evaluate_agent(
                    self.eval_envs,
                    self.agent,
                    n_steps=self.config.n_eval_steps,
                    device=self.config.device,
                    seed=self.config.seed + iteration,
                )
                eval_mean_rewards = np.nan if not eval_rewards else np.mean(eval_rewards)
                eval_mean_steps = np.nan if not eval_steps else np.mean(eval_steps)

                self.eval_rewards_hist.append(eval_mean_rewards)
                self.eval_rewards_steps.append(self.global_step)

                if self.wandb_log:
                    wandb.log(
                        {
                            "eval/mean_rewards": eval_mean_rewards,
                            "eval/mean_steps": eval_mean_steps,
                        },
                        step=self.global_step,
                    )
                self.last_eval = self.global_step

            end_time = time.time()
            print(
                f"Iter {iteration}/{self.config.n_iterations} took {end_time - start_time:.2f} seconds"
            )

        # Save model if configured
        if self.config.save_model:
            _ = save_model(
                self.agent,
                self.optimizer,
                self.train_envs,
                Path(__file__).parent / f"ppo_checkpoint_{self.config.env_name}.pt",
            )

        if self.wandb_log:
            wandb.finish()


def PPOTester(
    seed: int = 0,
    ckpt_path: Optional[Union[str, Path]] = None,
    n_episodes: int = 10,
    render: bool = False,
    agent_cls: type[nn.Module] = Agent,
    env_name: str = "DroneReachPos-v0", 
    test_env: VectorEnv = None,
    **kwargs: dict[str, Any]
):
    set_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_device = "cpu"
    n_envs = 1

    # Load checkpoint
    if ckpt_path is None:
        Path(__file__).parent / f"ppo_checkpoint_{env_name}.pt"
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint file not found: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if test_env is None:
        if not env_name == "DroneReachPos-v0":
            kwargs["n_samples"] = checkpoint.get("n_samples", 1)
        _, test_env = make_envs(env_name, n_envs, n_envs, env_device, **kwargs)
    test_env = RecordData(test_env)

    # Create agent and load state
    agent = agent_cls(test_env).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])

    # Test for n episodes
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = test_env.reset(seed=seed + episode)
        done = torch.zeros(n_envs, dtype=bool, device=device)
        episode_reward = 0
        steps = 0

        while not done.any():
            with torch.no_grad():
                action, _, _, _ = agent.action_and_value(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if render:
                test_env.render()

            done = terminated | truncated
            episode_reward += reward[0].item()
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    test_env.plot_eval()
    episode_rmse = test_env.calc_rmse()

    test_env.unwrapped.sim.close()

    print(
        f"\nAverage episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average episode standard tracking error: {episode_rmse:.3f} mm")

    return np.mean(episode_rewards), episode_rmse

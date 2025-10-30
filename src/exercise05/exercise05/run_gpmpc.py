import logging
import time
from collections import defaultdict

import crazyflow  # noqa: F401, register environments
import numpy as np
from exercise05.gpmpc import GPMPC
from exercise05.utils import obs_to_state, set_seed
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy


class Evaluator:
    """Class to evaluate MPC controllers on the Crazyflie environment."""

    def __init__(self, env: JaxToNumpy, ctrl: GPMPC, seed: int = 42, verbosity: str = "info"):
        self.logger = logging.getLogger(self.__class__.__name__)
        verbosity = verbosity.lower()
        self.logger.setLevel(getattr(logging, verbosity.upper(), "INFO"))
        self.env = env
        traj = env.unwrapped.trajectory
        self.traj = np.concatenate([traj, np.zeros((traj.shape[0], 9))], axis=-1)
        self.ctrl = ctrl
        self.seed = seed
        self.runs = {}
        set_seed(seed)

    def train_controller(self, ctrl: GPMPC, episode_data: dict):
        """Train the controller on the provided episode data. Assumes episode_data contains keys 'obs' and 'action', and than the controller has methods 'add_data' and 'fit'."""
        obs, actions = episode_data["obs"], episode_data["action"]
        next_obs = obs[1:, ...]
        obs = obs[:-1, ...]
        ctrl.add_data(obs, actions, next_obs)
        ctrl.fit() if hasattr(ctrl, "fit") else ctrl.train()

    def evaluate_epoch(self, ctrl: GPMPC, env_seed: int, max_steps: int = 5000, log_steps: int = 50) -> dict:
        """Evaluate the controller on the environment.

        Args:
            ctrl (GPMPC): The controller to evaluate.
            env_seed (int): Seed for the environment.
            max_steps (int): Maximum number of steps to run in the environment.
            log_steps (int): Log progress every `log_steps` steps.

        Returns:
            dict: A dictionary containing episode data.
        """
        if max_steps is None:
            max_steps = self.traj.shape[0] * 1.05
        episode_data = defaultdict(list)
        ctrl.reset()
        obs, _ = self.env.reset(seed=env_seed)
        obs = obs_to_state(obs)

        episode_data["obs"].append(obs)

        self.env.action_space.seed(env_seed)
        ctrl_data = defaultdict(list)
        inference_time_data = []
        step = 0
        while True:
            if step == 0:
                self.logger.debug(f"Starting new episode with env seed {env_seed}")
            time_start = time.perf_counter()
            action = ctrl.select_action(obs)
            inference_time_data.append(time.perf_counter() - time_start)
            # Vector environment expects a batched action (world size 1) in float32
            obs, reward, terminated, truncated, _ = self.env.step(action.astype(np.float32).reshape(1, -1))
            obs = obs_to_state(obs)
            done = terminated or truncated
            step_data = {"obs": obs, "action": action, "done": done, "reward": reward, "length": 1}
            for key, val in step_data.items():
                episode_data[key].append(val)

            step += 1
            if step % log_steps == 0:
                self.logger.debug(f"Step {step}, Reward: {reward}, Done: {done}")
            if done:
                break
            if step >= max_steps:  # safety break to avoid infinite loops
                self.logger.debug(f"Reached max steps {max_steps}, terminating episode.")
                break
        # Convert lists to arrays
        for key, val in episode_data.items():
            episode_data[key] = np.array(val)
        episode_data["ctrl_type"] = ctrl.ctrl_type if hasattr(ctrl, "ctrl_type") else ctrl.__class__.__name__
        episode_data["controller_data"] = dict(ctrl_data)
        episode_data["inference_time_data"] = inference_time_data
        return episode_data

    def run(self, n_epochs: int = 3, extra_train_runs: bool = True, print_stats: bool = True) -> dict:
        """Runs a MPC for multiple epochs and trains it in between if applicable.

        Args:
            n_epochs (int): Number of epochs to run.
            extra_train_runs (bool): If True, performs additional training runs with random seeds. If False, uses the data from the evaluation run for training.
            print_stats (bool): If True, prints timing statistics.
        """
        rng = np.random.default_rng(self.seed)
        assert n_epochs < int(1e3), f"Number of epochs must be less than 1e3, got {n_epochs}"
        train_seeds = rng.choice(np.iinfo(np.int32).max, size=int(1e6), replace=False)[: n_epochs + 1]
        eval_seed = train_seeds[-1]
        self.ctrl.reset(full_reset=True)
        for epoch in range(n_epochs):
            t1 = time.perf_counter()
            self.runs[epoch] = self.evaluate_epoch(self.ctrl, env_seed=int(eval_seed))
            t2 = time.perf_counter()
            t3 = time.perf_counter()
            if isinstance(self.ctrl, GPMPC) and epoch < n_epochs - 1:
                if extra_train_runs:
                    train_run = self.evaluate_epoch(self.ctrl, env_seed=int(train_seeds[epoch]))
                else:
                    train_run = self.runs[epoch]
                t3 = time.perf_counter()
                self.train_controller(self.ctrl, train_run)
            t4 = time.perf_counter()
            # Print timing table
            if print_stats:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
                print(f"Controller Type: {self.runs[epoch]['ctrl_type']}")
                print("\nExecution Times (seconds):")
                print(f"{'Operation':<25} {'Time (s)':<10}")
                print("-" * 35)
                # print(f"{'Evaluate GP':<25} {t4 - t3:>10.2f}")
                print(f"{'Evaluate Controller':<25} {t2 - t1:>10.2f}")
                print(
                    f"{'AVG Inference Time (ms)':<25} {1000 * np.mean(self.runs[epoch]['inference_time_data']):>10.2f}"
                )
                print(f"{'Train (next) Controller':<25} {t4 - t3:>10.2f}")

    def show_results(self, trajectory: bool = True, errors: bool = True):
        if trajectory:
            _ = self.plot_trajectory()
        if errors:
            errorss = self.compute_errors()
            print("\nTrajectory Tracking Errors (L2 norm):")
            print(f"{'Epoch':<10} {'Controller':<15} {'Mean Error':<15} {'Max Error':<15}")
            print("-" * 40)
            for epoch, (ctrl_type, error) in enumerate(errorss.items()):
                print(f"{epoch:<10} {ctrl_type:<15} {np.mean(error):<15.4f} {np.max(error):<15.4f}")

    def plot_trajectory(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 6))

        # x-z plane
        idx = [0, 2]
        plt.plot(self.traj[:, idx[0]], self.traj[:, idx[1]], label="Reference", color="gray", linestyle="-")
        for epoch, run in self.runs.items():
            plt.plot(
                run["obs"][: self.traj.shape[0], idx[0]],
                run["obs"][: self.traj.shape[0], idx[1]],
                label=f"{run['ctrl_type']} epoch %s" % epoch,
            )
        plt.title("X-Z plane path")
        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")
        plt.legend()
        plt.show()
        return fig

    def compute_errors(self):
        errors = {}
        for i, run in self.runs.items():
            max_steps = min(run["obs"].shape[0], self.traj.shape[0])
            obs = run["obs"][:max_steps, :3]
            traj = self.traj[:max_steps, :3]
            error = np.linalg.norm(obs - traj, axis=1)
            errors[f"{run['ctrl_type']} {i}"] = error
        return errors

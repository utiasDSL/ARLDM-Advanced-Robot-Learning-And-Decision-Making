from pathlib import Path
from typing import Any

import jax.numpy as jp
import numpy as np
from gymnasium.spaces import flatten_space
from gymnasium.vector import (
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
    VectorWrapper,
)
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray


class FlattenJaxObservation(VectorObservationWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        return jp.concatenate([v for v in observations.values()], axis=-1)

class AngleReward(VectorRewardWrapper):
    """Wrapper to penalize orientation in the reward."""

    def __init__(self, env: VectorEnv):
        super().__init__(env)

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        actions = actions.at[..., -1].set(0.0) # optional: block yaw output because we don't need it
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        # apply rpy penalty
        rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rewards -= 0.08 * rpy_norm
        return observations, rewards, terminations, truncations, infos
    
class RecordData(VectorWrapper):
    """Wrapper to record usefull data for debugging."""

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._record_act  = []
        self._record_pos  = []
        self._record_goal = []
        self._record_rpy  = []

    def step(self, actions: Any):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        raw_env = self.env.unwrapped

        act = np.asarray(actions)
        self._record_act.append(act.copy())

        pos = np.asarray(raw_env.sim.data.states.pos[:, 0, :])   # shape: (n_worlds, 3)
        self._record_pos.append(pos.copy())

        if type(raw_env).__name__ == "FigureEightEnv":
            goal = np.asarray(raw_env.trajectory[raw_env.steps.squeeze(1)])
        elif type(raw_env).__name__ == "ReachPosEnv":
            goal = np.asarray(raw_env._goal)
        elif type(raw_env).__name__ == "RandTrajEnv":
            goal = np.asarray(raw_env.trajectories[np.arange(raw_env.steps.shape[0]), raw_env.steps.squeeze(1)])
        else:
            goal = np.zeros_like(pos)
        self._record_goal.append(goal.copy())

        rpy = np.asarray(R.from_quat(raw_env.sim.data.states.quat[:, 0, :]).as_euler("xyz"))
        self._record_rpy.append(rpy.copy())

        return obs, rewards, terminated, truncated, infos
    
    def calc_rmse(self):
        # compute rmse for all worlds
        pos = np.array(self._record_pos)     # shape: (T, num_envs, 3)
        goal = np.array(self._record_goal)   # shape: (T, num_envs, 3)
        pos_err = np.linalg.norm(pos - goal, axis=-1)  # shape: (T, num_envs)
        rmse = np.sqrt(np.mean(pos_err ** 2))*1000 # mm

        return rmse
    
    def plot_eval(self, save_path: str = "eval_plot.png"):
        import matplotlib
        matplotlib.use("Agg")  # render to raster images
        import matplotlib.pyplot as plt
        actions = np.array(self._record_act)
        pos = np.array(self._record_pos)
        goal = np.array(self._record_goal)
        rpy = np.array(self._record_rpy)

        # Plot the actions over time
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
        for i in range(4):
            axes[i].plot(actions[:, 0, i])
            axes[i].set_title(f"{action_labels[i]} Command")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Action Value")
            axes[i].grid(True)

        # Plot position components
        position_labels = ["X Position", "Y Position", "Z Position"]
        for i in range(3):
            axes[4 + i].plot(pos[:, 0, i])
            axes[4 + i].set_title(position_labels[i])
            axes[4 + i].set_xlabel("Time Step")
            axes[4 + i].set_ylabel("Position (m)")
            axes[4 + i].grid(True)
        # Plot goal position components in same plots
        for i in range(3):
            axes[4 + i].plot(goal[:, 0, i], linestyle="--")
            axes[4 + i].legend(["Position", "Goal"])
        # Plot error in position
        pos_err = np.linalg.norm(pos[:, 0] - goal[:, 0], axis=1)
        axes[7].plot(pos_err)
        axes[7].set_title("Position Error")
        axes[7].set_xlabel("Time Step")
        axes[7].set_ylabel("Error (m)")
        axes[7].grid(True)

        # Plot angle components (roll, pitch, yaw)
        rpy_labels = ["Roll", "Pitch", "Yaw"]
        for i in range(3):
            axes[8 + i].plot(rpy[:, 0, i])
            axes[8 + i].set_title(f"{rpy_labels[i]} Angle")
            axes[8 + i].set_xlabel("Time Step")
            axes[8 + i].set_ylabel("Angle (rad)")
            axes[8 + i].grid(True)

        # compute RMSE for position
        rmse_pos = np.sqrt(np.mean(pos_err**2))
        axes[11].text(0.1, 0.5, f"Position RMSE: {rmse_pos*1000:.3f} mm", fontsize=14)
        axes[11].axis("off")

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / save_path)

        return fig, axes, rmse_pos
    
class RandTrajTestWrapper(VectorWrapper):
    """A wrapper that overwrites student's trajectories with stored random trajectories after reset."""

    def __init__(self, env: VectorEnv, trajectory: NDArray):
        super().__init__(env)
        # load trajectory
        self.sample_trajectory = trajectory

    def reset(self, **kwargs: dict[str, Any]):
        """Call student's reset, then replace trajectories with random spline trajectories."""
        obs, info = self.env.reset(**kwargs)

        trajectory = self.sample_trajectory
        unwrapped_env = self.env.unwrapped

        # Assign to student's env
        if hasattr(unwrapped_env, "trajectory"): # Single trajectory mode
            unwrapped_env.trajectory = trajectory
        elif hasattr(unwrapped_env, "trajectories"):
            unwrapped_env.trajectories = trajectory[None, :, :]
        else:
            raise AttributeError("Student environment has neither 'trajectory' nor 'trajectories'.")

        return obs, info
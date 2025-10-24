import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
import scipy
import torch
import yaml
from crazyflow.sim.physics import ang_vel2rpy_rates
from munch import munchify
from scipy.spatial.transform import Rotation as R


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discretize_dynamics_rk4(
    f: cs.Function, dt: float, x: cs.SX, u: cs.SX, p: cs.SX = None, as_function: bool = False
) -> Union[cs.Function, cs.SX]:
    """Discretize a continuous-time system using the Runge-Kutta 4 method.

    Args:
        f: Continuous-time dynamics function.
        dt: Time step for discretization.
        x: State variable.
        u: Control input variable.
        p: Optional parameter variable.
        as_function: If True, return a casadi Function, otherwise return the discretized state.
    """
    if p is None:
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        k1 = f(x, u, p)
        k2 = f(x + dt / 2 * k1, u, p)
        k3 = f(x + dt / 2 * k2, u, p)
        k4 = f(x + dt * k3, u, p)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if as_function:
        return cs.Function("f_disc", [x, u, p], [x_next])
    return x_next


def discretize_linear_system(
    A: np.ndarray, B: np.ndarray, dt: float, exact: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """(Non-exact) Discretization of a linear system.

    Args:
        A (np.ndarray): Continuous-time state matrix.
        B (np.ndarray): Continuous-time input matrix.
        dt (float): Time step.
        exact (bool): Whether to use exact discretization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Discrete-time state and input matrices.
    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        return Md[:state_dim, :state_dim], Md[:state_dim, state_dim:]

    return np.eye(state_dim) + A * dt, B * dt


def remove_path(path: str):
    """Remove a directory and its contents or a file if it exists."""
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def obs_to_state(obs: dict[str, npt.NDArray]) -> npt.NDArray:
    """Convert the observation from the obs dictionary to the state vector.

    Args:
        obs (dict[str, NDArray]): A dictionary containing observation data.
            Keys include:
                - "pos" (NDArray): Position vector, shape (3,).
                - "vel" (NDArray): Linear velocity vector, shape (3,).
                - "quat" (NDArray): Orientation in quaternion format, shape (4,),
                                    we use scalar-last order (x, y, z, w).
                - "rpy_rates" (NDArray): Angular velocity (roll, pitch, yaw rates), shape (3,).

    Returns:
        NDArray: A single state vector containing concatenated observation data.
    """
    # Extract position
    pos = obs["pos"].squeeze()  # shape: (3,)

    # Extract linear velocity
    vel = obs["vel"].squeeze()  # shape: (3,)

    # Extract orientation as quaternion and convert to Euler angles
    quat = obs["quat"].squeeze()  # shape: (4,)
    euler = R.from_quat(quat).as_euler("xyz")  # shape: (3,), Euler angles
    # euler = euler[::-1] # [roll, pitch, yaw]
    # Extract angular velocity
    rpy_rates = ang_vel2rpy_rates(obs["ang_vel"].squeeze(), quat)  # shape: (3,)

    # Concatenate into a single state vector
    state = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], *euler, *rpy_rates])

    return state


##################### path and config utils #####################


def setup_config_and_result_dir(config_name: str = "base_config_hidden.yaml") -> dict:
    """Load the configuration file and setup the results export folder.

    Args:
        config_name: Name of the configuration file

    Returns:
        config: the loaded configuration
    """
    config_path = Path(__file__).parent / config_name
    if not config_path.exists():
        root_dir = str(Path(__file__).parents[2])
        config_name = config_name.removesuffix(".yaml")
        config_name += "_private.yaml"
        config_path = os.path.join(root_dir, "test", "configs", config_name)

    # root of the project
    # config_path = Path(__file__).parent / "config" / config_name

    with open(config_path, "r") as file:
        config = munchify(yaml.safe_load(file))

    save_dir = Path(__file__).parents[1] / config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = save_dir
    return config


def mkdir_date(path: Path) -> Path:
    """Make a unique directory within the given directory with the current time as name.

    Args:
        path: Parent folder path

    Returns:
        the path for storing results
    """
    # assert path.is_dir(), f"Path {path} is not a directory"
    save_dir = path / datetime.now().strftime("%Y_%m_%d_%H_%M")
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        t = 1
        while save_dir.is_dir():
            curr_date_unique = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})"
            save_dir = path / (curr_date_unique)
            t += 1
        save_dir.mkdir(parents=True)
    return save_dir

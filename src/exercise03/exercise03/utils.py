import numpy as np
import scipy
import scipy.linalg
from crazyflow.sim.physics import ang_vel2rpy_rates
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def obs_to_state(obs: dict[str, NDArray]) -> NDArray:
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
    state = np.concatenate([pos, euler, vel, rpy_rates])

    return state


def discretize_linear_system(A, B, dt, exact=False):
    """Discretize a linear system.

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix.
        B: np.array, input matrix.
        dt: scalar, step time interval.
        exact: bool, if to use exact discretization.

    Returns:
        Discretized matrices Ad, Bd.
    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)  # noqa: E741
        Ad = I + A * dt
        Bd = B * dt
    return Ad, Bd

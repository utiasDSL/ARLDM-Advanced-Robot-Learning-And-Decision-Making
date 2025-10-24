import numpy as np
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
                - "ang_vel" (NDArray): Angular velocity (roll, pitch, yaw rates), shape (3,).

    Returns:
        NDArray: A single state vector containing concatenated observation data.
    """
    ########################################################################
    # TODO:
    # Convert the observation from the obs dictionary to the state vector.
    # Hints:
    #   - Convert quaternion to Euler angles (in radians) in "xyz" order.
    #     - Rotation.from_quat:
    #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    #     - Rotation.as_euler:
    #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html
    #   - Use `ang_vel2rpy_rates()` to convert `ang_vel` into `rpy_rates`
    ########################################################################
    



















    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    return state


def discretize_linear_system(A: NDArray, B: NDArray, dt: float) -> tuple[NDArray, NDArray]:
    """Discretization of a linear system. (Euler Approximation).

    dx/dt = A x + B u --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: Linear system transition matrix.
        B: Linear input matrix.
        dt: Step time interval.

    Returns:
        The discrete linear state matrix Ad and the discrete linear input matrix Bd.
    """
    ########################################################################
    # TODO: Discretization of a linear system.
    # Hints:
    # - It's sufficient to use First-order(Euler) approximation.
    ########################################################################
    








    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    return Ad, Bd

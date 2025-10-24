import jax.numpy as jnp
import numpy as np
from crazyflow.constants import GRAVITY, MASS
from jax.scipy.spatial.transform import Rotation as R

try:
    from base_controller import BaseController
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # required for importing base_controller in development repository
    from base_controller import BaseController


class PDDrone(BaseController):
    def __init__(
        self,
        des_pos,
        des_vel=np.zeros(3),
        des_yaw=np.zeros(1),
        kp=np.array([0.4, 0.4, 1.25]),
        kd=np.array([0.2, 0.2, 0.5]),
        drone_mass=MASS,
    ):
        """Initializes the PD controller with desired positions, velocities, yaw, and control gains.

        Parameters:
            des_pos (array-like): Desired position as a 3-element array [x, y, z].
            des_vel (array-like, optional): Desired velocity as a 3-element array [vx, vy, vz]. Defaults to a zero vector.
            des_yaw (array-like, optional): Desired yaw as a 1-element array [yaw]. Defaults to a zero vector.
            kp (array-like, optional): Proportional gain as a 3-element array [kx, ky, kz]. Defaults to [0.4, 0.4, 1.25].
            kd (array-like, optional): Derivative gain as a 3-element array [kx, ky, kz]. Defaults to [0.2, 0.2, 0.5].
        """
        super().__init__()
        self.kp = kp
        self.kd = kd
        self.des_pos = des_pos
        self.des_vel = des_vel
        self.des_yaw = des_yaw
        self.drone_mass = drone_mass

    def step_control(self, pos, vel, quat):
        # refer to the lecture notes that explains the PD controller: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/#mellinger-controller
        pos, vel, quat = jnp.array(pos), jnp.array(vel), jnp.array(quat)
        pos_error, vel_error = self.des_pos - pos, self.des_vel - vel
        # Compute target thrust
        thrust = self.kp * pos_error + self.kd * vel_error
        thrust = thrust.at[2].add(self.drone_mass * GRAVITY)
        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(quat).as_matrix()[..., 2].T
        # Project the thrust onto the z-axis
        thrust_desired = np.clip(
            thrust @ z_axis, 0.3 * self.drone_mass * GRAVITY, 1.8 * self.drone_mass * GRAVITY
        )
        # Update the desired z-axis
        z_axis = thrust / np.linalg.norm(thrust)
        yaw_axis = np.concatenate([np.cos(self.des_yaw), np.sin(self.des_yaw), np.array([0.0])])
        y_axis = np.cross(z_axis, yaw_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        euler_desired = R.from_matrix(np.vstack([x_axis, y_axis, z_axis]).T).as_euler("xyz")
        return np.concatenate([np.atleast_1d(thrust_desired), euler_desired])

import numpy as np
import scipy.linalg
from exercise02.utils import (
    discretize_linear_system,
)  # absolute import required for running tests in development repository
from numpy import ndarray

try:
    from base_controller import BaseController
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # required for importing base_controller in development repository
    from base_controller import BaseController


class LQR(BaseController):
    """Linear quadratic regulator."""

    def __init__(
        self,
        env,
        # Model args.
        x_eq,
        u_eq,
        Q,
        R,
        discrete_dynamics: bool = True,
    ):
        """Initializes the task and controller.

        Args:
            env: The task or environment to control.
            x_eq: The equilibrium state of the system.
            u_eq: The equilibrium input of the system.
            Q: State cost weight matrix (must be positive semi-definite).
            R: Input cost weight matrix (must be positive definite).
            discrete_dynamics (bool): Specifies whether to use discrete-time dynamics.
                                    (In this exercise, only discrete dynamics are used.)
        """
        super().__init__()

        self.env = env
        # Controller params.
        self.model = self.get_symbolic(self.env)
        self.discrete_dynamics = discrete_dynamics
        self.x_eq = x_eq
        self.u_eq = u_eq
        self.Q = Q
        self.R = R

        self.gain = self.compute_lqr_gain(self.model, self.x_eq, self.u_eq, self.Q, self.R)

    @staticmethod
    def compute_lqr_gain(model, x_0, u_0, Q, R) -> ndarray:
        """Computes the LQR gain from the model.

        Args:
            model (SymbolicModel): The SymbolicModel CasADi of the system.
            x_0 (ndarray): The linearization point of the state X.
            u_0 (ndarray): The linearization point of the input U.
            Q (ndarray): The state cost matrix Q.
            R (ndarray): The input cost matrix R.

        Returns:
            gain (ndarray, shape: (n_u, n_x)): The LQR gain for the system,
                where n_u and n_x are the dimension of state and input.
        """
        ########################################################################
        # TODO: Linearize the system dynamic matrices A and B around the
        #       operating point. Hint: inspect the previously mentioned
        #       setup_linearization() method.  Use self.model.df_func to get
        #       the continuous-time dynamics.
        ########################################################################
        






        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################

        ########################################################################
        # TODO: Implement discrete-time LQR: Compute controller gain.
        ########################################################################
        








        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return gain

    def step_control(self, obs, goal, step=None, task="stabilization"):
        """Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            goal (ndarray or list[ndarray]): The goal state(s).
                - For 'stabilization': a single fixed goal state.
                - For 'tracking': a list or array of goal states over time.
            step (int): The current timestep.
            task (str): The control task type, either 'stabilization' or 'tracking'.
                - 'stabilization': Stabilize the system at a fixed goal state.
                - 'tracking': Track a time-varying goal trajectory.

        Returns:
            action (ndarray, shape (1, n_u)): The action chosen by the controller,
                where n_u represents the dimension of the control input u.
        """
        ########################################################################
        # TODO:
        # 1. Implement the control logic for 'stabilization'.
        # 2. For 'stabilization', compute the control input using the
        #    difference between the current observation `obs` and the
        #    fixed goal `goal`. Use the precomputed LQR gain (`self.gain`).
        # 3. Don't forget to add the equilibrium control input `self.u_eq`
        #    in both cases to offset any biases in the system.
        # 4. Clip the control input to ensure it stays within the action space
        #    bounds defined by the environment.
        # 5. Convert the control input to the correct data type (`float32`)
        #    for compatibility with the environment.
        ########################################################################
        













        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################

        return control_input

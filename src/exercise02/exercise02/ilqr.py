"""Iterative Linear Quadratic Regulator (ILQR)."""

import numpy as np
from crazyflow.constants import GRAVITY, MASS
from exercise02.utils import discretize_linear_system, obs_to_state
from termcolor import colored

try:
    from base_controller import BaseController
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # required for importing base_controller in development repository
    from base_controller import BaseController


class ILQR(BaseController):
    """Iterative linear quadratic regulator."""

    def __init__(
        self,
        env,
        # Model args
        Q,
        R,
        goal,
        input_ff=None,
        gains_fb=None,
        discrete_dynamics: bool = True,
        # iLQR args
        max_iterations: int = 10,
    ):
        """Creates task and controller.

        Args:
            env: The task/environment.
            Q (list or np.array): Diagonals of the state cost weight matrix.
            R (list or np.array): Diagonals of the control cost weight matrix.
            goal (np.array): Target state for the controller.
            input_ff (np.array, optional): Feedforward input, if any.
            gains_fb (np.array, optional): Feedback gains, if any.
            discrete_dynamics (bool, optional): Whether to use discrete-time dynamics. Defaults to True.
            max_iterations (int, optional): Number of iterations for iLQR optimization. Defaults to 15.
        """
        super().__init__()

        # Model parameters
        self.discrete_dynamics = discrete_dynamics

        # iLQR parameters
        self.max_iterations = max_iterations

        self.env = env

        # Controller params.
        self.model = self.get_symbolic(self.env)

        self.Q = Q
        self.R = R
        self.goal = goal

        self.nx = self.Q.shape[0]
        self.nu = self.R.shape[0]

        self.u_eq = np.array([MASS * GRAVITY, 0, 0, 0])

        # initialize gain
        self.input_ff = np.copy(input_ff) if input_ff is not None else np.zeros((self.nu, 2500))
        self.gains_fb = (
            np.copy(gains_fb) if gains_fb is not None else np.zeros((2500, self.nu, self.nx))
        )

        # Control stepsize.
        self.stepsize = self.model.dt

        self.ite_counter = 0
        self.input_ff_best = None
        self.gains_fb_best = None

        self.collision = False

        self.lamb = 1

    def step_control(self, state_curr, step: int):
        theta_fb = self.gains_fb_best[step]
        theta_ff = self.input_ff_best[:, step]
        control_input = theta_ff + theta_fb.dot(state_curr)
        return control_input

    def learn(self):
        self.lamb = 1
        lamb_factor = 10  # The amount for which to increase lambda when training fails.
        lamb_max = 1000  # The maximum lambda allowed.
        epsilon = 0.01  # The convergence tolerance.
        iter = 0

        previous_total_cost = -float("inf")

        while (
            iter <= self.max_iterations
        ):  # and (np.linalg.norm(np.squeeze(duff)) > norm_threshold or iter == 0):
            # Forward / "rollout" of the current policy
            obs, _ = self.env.reset(seed=42)
            state = obs_to_state(obs)  # (12,)

            total_cost = 0
            for step in range(2500):
                # Calculate control input.
                control_input = self.input_ff[:, step] + self.gains_fb[step].dot(state)

                # Clip the control input to the specified range
                control_input = np.clip(
                    control_input, self.env.action_space.low, self.env.action_space.high
                )

                # Reshape and Convert to np.ndarray
                action = control_input.reshape(1, 4).astype(np.float32)  # (1, 4)

                # Save rollout data.
                if step == 0:
                    # Initialize state and input stack.
                    state_stack = state
                    input_stack = action
                else:
                    # Save state and input.
                    state_stack = np.vstack((state_stack, state))  # (N, 12)
                    input_stack = np.vstack((input_stack, action))  # (N, 4)

                # Step forward.
                obs, _, terminated, _, _ = self.env.step(action)
                state = obs_to_state(obs)  # (12,)

                if terminated:
                    self.collision = True

                loss_k = self.model.loss(
                    x=state, u=action.reshape(-1), Xr=self.goal, Ur=self.u_eq, Q=self.Q, R=self.R
                )
                cost = loss_k["l"].toarray()  # cost
                total_cost += cost

            self.env.close()

            print(colored(f"Iteration: {iter}, Cost: {total_cost}", "green"))
            print(colored("--------------------------", "green"))

            if iter == 0 and self.collision:
                print(
                    colored(
                        "[ERROR] The initial policy might be unstable. "
                        + "Break from iLQR updates.",
                        "red",
                    )
                )
                break

            delta_cost = total_cost - previous_total_cost
            if iter == 0:
                # Save best iteration.
                self.best_iter = iter
                self.best_cost = total_cost
                previous_total_cost = total_cost
                self.input_ff_best = np.copy(self.input_ff)
                self.gains_fb_best = np.copy(self.gains_fb)

                # Initialize improved flag.
                prev_ite_improved = False

            elif delta_cost > 0.0:
                # If cost is increased, increase lambda
                self.lamb *= lamb_factor

                print(
                    f"Cost increased by {delta_cost}. "
                    + "Set feedforward term and controller gain to that "
                    "from the previous iteration. "
                    f"Increased lambda to {self.lamb}."
                )
                print(f"Current policy is from iteration {self.best_iter}.")

                # Reset feedforward term and controller gain to that from
                # the previous iteration.
                self.input_ff = np.copy(self.input_ff_best)
                self.gains_fb = np.copy(self.gains_fb_best)

                # Set improved flag to False.
                prev_ite_improved = False

                # Break if maximum lambda is reached.
                if self.lamb > lamb_max:
                    print(colored("Maximum lambda reached.", "red"))
                    self.lamb = lamb_max

                iter += 1
                continue

            elif delta_cost <= 0.0:
                # Save feedforward term and gain.
                self.best_iter = iter
                self.best_cost = total_cost
                previous_total_cost = total_cost
                self.input_ff_best = np.copy(self.input_ff)
                self.gains_fb_best = np.copy(self.gains_fb)

                # Check consecutive cost increment (cost decrement).
                if abs(delta_cost) < epsilon and prev_ite_improved:
                    # Cost converged.
                    print(
                        colored(
                            "iLQR cost converged with a tolerance " + f"of {epsilon}.", "yellow"
                        )
                    )
                    break
                # Set improved flag to True.
                prev_ite_improved = True

            ####################################################################
            # TODO (1): Initialize the backward pass.
            #   - Retrieve the terminal state (from state_stack[-1]) and
            #     reshape.
            #   - Compute the terminal cost parameters (s, Sv, Sm) using
            #     self.terminal_cost_quad(...) .
            ####################################################################
            






            ####################################################################
            #                           END OF YOUR CODE
            ####################################################################

            ####################################################################
            # TODO (2): Implement the backward pass (DP approach).
            #   - Reverse iterate over the time steps (e.g., range(2500)).
            #   - For each step k:
            #       1) Extract the current state (state_k) and input (input_k).
            #       2) Obtain the linearized dynamics Ad_k, Bd_k by calling
            #          self.dynamic_lin_dics(...).
            #       3) Compute the stage cost derivatives via
            #          self.stage_cost_quad
            #       4) Update the policy by calling self.update_policy(...).
            #       5) Store the resulting feedforward and feedback gains
            #          (theta_ff, theta_fb) in self.input_ff and self.gains_fb.
            ####################################################################
            # "Backward pass": Calculate the coefficients (s,Sv,Sm) for the
            # value functions at earlier times by proceeding backwards in time
            # (DP-approach)
            




















            ####################################################################
            #                           END OF YOUR CODE
            ####################################################################

            iter += 1

    def dynamic_lin_disc(self, x_e, u_e):
        """Computes the linearized dynamics around the operating point (x_e, u_e) and discretizes the system.

        Args:
            x_e (np.ndarray): The operating point for the state, where the system is linearized. Shape: (12,)
            u_e (np.ndarray): The operating point for the control input, where the system is linearized. Shape: (4,)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Ad_k (np.ndarray): Discretized state transition matrix. Shape: (12, 12)
                - Bd_k (np.ndarray): Discretized input matrix. Shape: (12, 4)
        """
        ########################################################################
        # TODO:
        # Design a function to get discretized linear system dynamic matrices.
        # Hints:
        #   - Use self.model.df_func to get the continuous-time dynamics.
        #   - Don't forget discretization.
        ########################################################################
        







        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return Ad_k, Bd_k

    def terminal_cost_quad(self, x_N):
        """Computes the quadratic approximation of the terminal cost at the final state x_N.

        Args:
            x_N (np.ndarray): The terminal state. Shape: (12,)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - q_N (np.ndarray): The scalar terminal cost. Shape: (1,1)
                - Qv_N (np.ndarray): The gradient of the terminal cost with respect to the state (∂l/∂x). Shape: (12,1)
                - Qm_N (np.ndarray): The Hessian of the terminal cost with respect to the state (∂²l/∂x²). Shape: (12,12)
        """
        ########################################################################
        # TODO:
        #  Compute the loss at the terminal state (x_N). Use the model's
        # `loss` method to get the cost value, gradient, and Hessian.
        # Hints:
        #   - Convert each component to an array using `.toarray()`.
        #   - Check the code of `setup_linearization` in sim/symbolic.py
        ########################################################################
        







        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return q_N, Qv_N, Qm_N

    def stage_cost_quad(self, x_curr, u_curr):
        """Computes the stage cost and its quadratic approximation at the current state and control input.

        Args:
            x_curr (np.ndarray): The current state of the system. Shape: (12,)
            u_curr (np.ndarray): The current control input. Shape: (4,)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - q (np.ndarray): The scalar stage cost. Shape: (1,1)
                - Qv (np.ndarray): The gradient of the stage cost w.r.t. state (∂l/∂x). Shape: (12,1)
                - Qm (np.ndarray): The Hessian of the stage cost w.r.t. state (∂²l/∂x²). Shape: (12,12)
                - Rv (np.ndarray): The gradient of the stage cost w.r.t. control input (∂l/∂u). Shape: (4,1)
                - Rm (np.ndarray): The Hessian of the stage cost w.r.t. control input (∂²l/∂u²). Shape: (4,4)
                - Pm (np.ndarray): The cross-term Hessian of the stage cost w.r.t. state and input (∂²l/∂x∂u). Shape: (4,12)
        """
        ########################################################################
        # TODO:
        # Compute the loss, gradient, and Hessian at the current state and
        # input (x_curr,u_curr) using the model's `loss` method.
        # Hints:
        #   - Check the code of `setup_linearization` in sim/symbolic.py
        #   - Convert each component to an array using `.toarray()`.
        ########################################################################
        













        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return q, Qv, Qm, Rv, Rm, Pm

    def update_policy(
        self, Ad_k, Bd_k, q, Qv, Qm, Rv, Rm, Pm, s_next, Sv_next, Sm_next, x_curr, u_curr
    ):
        """Updates the policy based on the quadratic approximation of the value function.

        Args:
            Ad_k (np.ndarray): State transition matrix (discretized dynamics for state). Shape: (12, 12)
            Bd_k (np.ndarray): Input matrix (discretized dynamics for control input). Shape: (12, 4)
            q (np.ndarray): Scalar stage cost. Shape: (1,1)
            Qv (np.ndarray): Gradient of stage cost w.r.t. state (∂l/∂x). Shape: (12,1)
            Qm (np.ndarray): Hessian of stage cost w.r.t. state (∂²l/∂x²). Shape: (12,12)
            Rv (np.ndarray): Gradient of stage cost w.r.t. input (∂l/∂u). Shape: (4,1)
            Rm (np.ndarray): Hessian of stage cost w.r.t. input (∂²l/∂u²). Shape: (4,4)
            Pm (np.ndarray): Cross-term Hessian of stage cost w.r.t. state and input (∂²l/∂x∂u). Shape: (4,12)
            s_next (np.ndarray): Scalar value of the next state's cost-to-go function. Shape: (1,1)
            Sv_next (np.ndarray): Gradient of the next state's cost-to-go function w.r.t. state. Shape: (12,1)
            Sm_next (np.ndarray): Hessian of the next state's cost-to-go function w.r.t. state. Shape: (12,12)
            x_curr (np.ndarray): Current state. Shape: (12,)
            u_curr (np.ndarray): Current control input. Shape: (4,)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - theta_ff (np.ndarray): Feedforward control term. Shape: (4,)
                - theta_fb (np.ndarray): Feedback gain matrix. Shape: (4,12)
                - s (np.ndarray): Scalar value of the current state's cost-to-go function. Shape: (1,1)
                - Sv (np.ndarray): Gradient of the current state's cost-to-go function w.r.t. state. Shape: (12,1)
                - Sm (np.ndarray): Hessian of the current state's cost-to-go function w.r.t. state. Shape: (12,12)
        """
        ########################################################################
        # TODO:
        # Update the control law and coefficients of the approximation of the
        # value function at time step k.
        # Hints:
        #   - If you forget the theories and formulas, please refer to the
        #     lecture slides from the theory class.
        #   - Don't delete or change the trick which make sure H is
        #     well-conditioned for inversion.
        # NOTE:
        # Here, we use the equations in the lecture notes (script) as
        # reference implementation, and not the slides. You should also use the
        # equations from the lecture notes.
        ########################################################################
        








        ########## Don't delete this part (start)  ##########
        # Trick to make sure H is well-conditioned for inversion
        # 'H' here is the Hessian of the cost-to-go function with respect to
        # the control input (u).
        # It includes terms from the stage cost's own Hessian (Rm) and the
        # influence of the state Hessian (Sm_next) through the discretized
        # dynamics (Bd_k).
        # NOTE make sure to keep the naming of "H" consistent with your
        # code
        if not (np.isinf(np.sum(H)) or np.isnan(np.sum(H))):
            H = (H + H.transpose()) / 2
            H_eval, H_evec = np.linalg.eig(H)
            H_eval[H_eval < 0] = 0.0
            H_eval += self.lamb
            H_inv = np.dot(H_evec, np.dot(np.diag(1.0 / H_eval), H_evec.T))
        ########## Don't delete this part (end)  ##########
        
























        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return theta_ff, theta_fb, s, Sv, Sm

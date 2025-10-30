import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver  # type: ignore
from crazyflow.constants import GRAVITY, MASS
from crazyflow.control.control import MAX_THRUST, MIN_THRUST
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.symbolic import SymbolicModel, symbolic_attitude
from exercise05.mpc_utils import OcpRegistry
from exercise05.utils import (
    discretize_dynamics_rk4,
    discretize_linear_system,
    recursive_update,
    remove_path,
)

# Cases due to development repository structure
try:
    from base_controller import BaseController  # type: ignore
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from base_controller import BaseController  # type: ignore


@dataclass
class MPCConfig:
    state_mapping: dict = field(
        default_factory=lambda: {
            "x": 0,
            "y": 1,
            "z": 2,
            "phi": 3,
            "theta": 4,
            "psi": 5,
            "dx": 6,
            "dy": 7,
            "dz": 8,
            "dphi": 9,
            "dtheta": 10,
            "dpsi": 11,
        }
    )
    action_mapping: dict = field(default_factory=lambda: {"thrust": 0, "phi_cmd": 1, "theta_cmd": 2, "yaw_cmd": 3})
    U_EQ: np.ndarray = field(default_factory=lambda: np.array([GRAVITY * MASS, 0.0, 0.0, 0.0], dtype=np.float32))
    X_EQ: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    X_LOW: np.ndarray = field(
        default_factory=lambda: np.array([-2, -2, 0.05, -np.pi, -np.pi, -np.pi, -15, -15, -15, -10, -10, -10])
    )
    X_HIGH: np.ndarray = field(default_factory=lambda: np.array([2, 2, 3, np.pi, np.pi, np.pi, 15, 15, 15, 10, 10, 10]))
    U_LOW: np.ndarray = field(
        default_factory=lambda: np.array([4 * MIN_THRUST, -np.pi / 2, -np.pi / 2, -np.pi / 2], dtype=np.float32)
    )
    U_HIGH: np.ndarray = field(
        default_factory=lambda: np.array([4 * MAX_THRUST, np.pi / 2, np.pi / 2, np.pi / 2], dtype=np.float32)
    )
    horizon: int = 100
    solver_options: dict = field(
        default_factory=lambda: {
            "qp_solver": "PARTIAL_CONDENSING_HPIPM",
            "hessian_approx": "GAUSS_NEWTON",
            "integrator_type": "DISCRETE",
            "nlp_solver_type": "SQP",
            "nlp_solver_max_iter": 50,
            "tol": 1e-5,
        }
    )
    cost_params: dict = field(
        default_factory=lambda: {
            "Q": [100, 0.1, 100, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
            "R": [4, 0.5, 0.5, 0.5],
        }
    )


class MPC(BaseController):
    """Model Predictive Controller using acados for quadrotor attitude and position control.

    Args:
        env (DroneEnv): The Crazyflie environment.
        prior_params (dict): Parameters for the symbolic model (can differ from env params for model mismatch).
        output_dir (Path): Directory to save generated code and JSON files.
        cfg (dict, optional): Configuration dictionary for MPC setup.
        verbosity (str, optional): Logging verbosity level ("info", "debug", "warning").
        defaults (MPCDefaults, optional): Default parameters and mappings for the controller.

    Methods:
        reset(full_reset: bool = False): Reset the controller and optionally redeploy acados code.
        select_action(x0: np.ndarray) -> np.ndarray: Select action based on current state.
        step_control(...): Internal step for solving the OCP and returning the control action.
        reference_trajectory() -> np.ndarray: Get the reference trajectory for the current horizon.
        setup_prior_dynamics(): Linearize and discretize the prior model for LQR terminal cost.
        setup_ocp(override: bool = False): Build and deploy the acados OCP.
        setup_acados_model(...): Build the acados model from the symbolic model.
        setup_acados_ocp(...): Configure the acados OCP problem.
        prepare_for_solve(...): Set up solver state, references, and warm start for the current step.
    """

    mpc_cfg = MPCConfig()
    prior_params: dict = {
        "a": 20.9,
        "b": 3.6,
        "ra": -130,
        "rb": -16.3,
        "rc": 119.3,
        "pa": -100.0,
        "pb": -13.3,
        "pc": 84.47,
        "ya": -0.01,
        "yb": 0.0,
        "yc": 0.0,
    }

    def __init__(
        self,
        env: DroneEnv,
        output_dir: Path,
        prior_params: dict = {},
        cfg: dict = {},
        verbosity: str = "info",
        **kwargs: Any,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        assert verbosity.lower() in ["info", "debug", "warning", "error"], (
            "verbosity must be 'info', 'debug' or 'warning', 'error'"
        )
        self.logger.setLevel(getattr(logging, verbosity.upper()))
        self.verbosity = verbosity
        self.prior_params.update(prior_params)  # Update defaults with user params

        recursive_update(self.mpc_cfg, cfg)  # Update defaults with user config

        # Unpack environment and symbolic model
        self.env = env
        self.symbolic_model: SymbolicModel = symbolic_attitude(dt=1 / env.freq, params=self.prior_params)
        self.nx = self.symbolic_model.nx
        self.nu = self.symbolic_model.nu
        self.dt = self.symbolic_model.dt
        self.traj = env.trajectory

        # Pad trajectory with zeros if needed
        if self.traj.shape[1] != self.nx:
            missing = self.nx - self.traj.shape[1]
            self.traj = np.hstack((self.traj, np.zeros((self.traj.shape[0], missing))))

        # Unpack configuration
        self.T = self.mpc_cfg.horizon
        self.U_EQ = self.mpc_cfg.U_EQ
        self.X_EQ = self.mpc_cfg.X_EQ
        self.X_LOW = self.mpc_cfg.X_LOW
        self.X_HIGH = self.mpc_cfg.X_HIGH
        self.U_LOW = env.action_space.low.copy().squeeze() if hasattr(env.action_space, "low") else self.mpc_cfg.U_LOW
        self.U_HIGH = (
            env.action_space.high.copy().squeeze() if hasattr(env.action_space, "high") else self.mpc_cfg.U_HIGH
        )
        self.u_ref = np.repeat(self.U_EQ[..., None], self.T, axis=-1).T

        # Code creation variables
        self._output_dir = output_dir
        self._json_file = str(self._output_dir / "mpc_ocp.json")
        self._code_export_dir = str(self._output_dir / "mpc_c_generated_code")
        self._deployed = False

        self.prior_A, self.prior_B, self.lqr_gain = self.setup_prior_dynamics(self.mpc_cfg.cost_params)

        self.ctrl_type = "MPC"

        MPC.reset(self, full_reset=True)

    def reset(self, full_reset: bool = False):
        """Reset the controller state.

        Args:
            full_reset (bool): If True, redeploys the acados code..
        """
        self.traj_step = 0
        self.u_last = self.U_EQ.copy()
        self.x_sol_last = None
        self.u_sol_last = None
        if full_reset:
            self.ocp, self.ocp_solver, self.acados_model = self.setup_ocp(override=True)
        else:
            self.ocp_solver.reset()

    def setup_ocp(self, override: bool = True) -> AcadosOcp:
        """Build and deploy the acados OCP problem.

        Args:
            override (bool): If True, force redeployment.

        Returns:
            tuple: (ocp, ocp_solver, acados_model)
        """
        if self._deployed and not override:
            return self.ocp, self.ocp_solver, self.acados_model
        # Remove old files if they exist
        self.ocp = None
        self.ocp_solver = None
        self.acados_model = None
        self.registry = OcpRegistry(horizon=self.T)
        remove_path(self._json_file)  # Remove old json file if it exists
        remove_path(self._code_export_dir)  # Remove old code directory if it exists

        # Create new ocp
        acados_model = self.setup_acados_model(name="mpc", symbolic_model=self.symbolic_model, cfg=self.mpc_cfg)
        ocp = self.setup_acados_ocp(acados_model, cfg=self.mpc_cfg)

        def box(sym: cs.MX, low, high):
            A = np.vstack((-np.eye(low.shape[0]), np.eye(low.shape[0])))
            b = np.hstack((-low, high))
            return A @ sym - b

        self.registry.add_constraint(
            "state",
            box(acados_model.x, self.X_LOW, self.X_HIGH),
            ["path", "terminal"],
            allow_slack=False,
            l2_slack_penalty=1e5,
        )
        self.registry.add_constraint(
            "input",
            box(acados_model.u, self.U_LOW, self.U_HIGH),
            ["initial", "path"],
            allow_slack=False,
            l2_slack_penalty=1e7,
        )
        self.registry.build_parameters(ocp)
        self.registry.build_constraints(ocp)
        ocp_solver = AcadosOcpSolver(ocp, json_file=self._json_file, verbose=False)
        self._deployed = True
        return ocp, ocp_solver, acados_model

    def setup_acados_model(self, name: str, symbolic_model: SymbolicModel, cfg: dict = {}) -> AcadosModel:
        """Set up the acados model from the symbolic model.

        Args:
            name (str): Name of the model.
            symbolic_model (SymbolicModel): The symbolic model instance.
            cfg (dict): Configuration dictionary.

        Returns:
            acados_model: Configured AcadosModel instance.
        """
        if dataclasses.is_dataclass(cfg):
            cfg = dataclasses.asdict(cfg)
        acados_model = AcadosModel()
        # Core symbols
        acados_model.x = symbolic_model.x_sym
        acados_model.u = symbolic_model.u_sym

        # (1) Continuous-time explicit dynamics
        acados_model.f_expl_expr = symbolic_model.x_dot

        # (2) Continuous-time implicit dynamics (optional; unused in DISCRETE mode)
        xdot = cs.MX.sym("xdot", symbolic_model.nx, 1)
        acados_model.xdot = xdot
        acados_model.f_impl_expr = xdot - acados_model.f_expl_expr

        # (3) Discrete dynamics with Runge-Kutta 4 integration
        x_next = discretize_dynamics_rk4(
            symbolic_model.fc_func, symbolic_model.dt, acados_model.x, acados_model.u, as_function=False
        )
        acados_model.disc_dyn_expr = x_next

        # (4) Cost expressions
        acados_model.y_expr = cs.vertcat(acados_model.x, acados_model.u)  # Lagrange term
        acados_model.y_expr_e = acados_model.x  # Mayer term

        # Meta
        acados_model.name = name
        acados_model.t_label = "time"
        return acados_model

    def setup_acados_ocp(self, model: AcadosModel, cfg: dict = {}) -> AcadosOcp:
        """Set up the acados OCP problem.

        Args:
            model (AcadosModel): The AcadosModel instance.
            cfg (dict): Configuration dictionary for MPC setup.

        Returns:
            AcadosOcp: Configured AcadosOcp instance.
        """
        ocp = AcadosOcp()
        ocp.model = model
        nx, nu = self.nx, self.nu
        ny = nx + nu
        ny_e = nx
        if dataclasses.is_dataclass(cfg):
            cfg = dataclasses.asdict(cfg)

        solver_options = cfg.get("solver_options", self.mpc_cfg.solver_options)
        cost_params = cfg.get("cost_params", self.mpc_cfg.cost_params)
        Q = np.diag(cost_params.get("Q", self.mpc_cfg.cost_params["Q"]))
        R = np.diag(cost_params.get("R", self.mpc_cfg.cost_params["R"]))
        # Configure costs
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx : (nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # Placeholder y_ref, y_ref_e and initial state constraint. We update yref in select_action.
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.constraints.x0 = np.zeros(nx)

        # Set up solver options
        ocp.solver_options.N_horizon = self.T
        ocp.solver_options.qp_solver = solver_options.get("qp_solver", self.mpc_cfg.solver_options["qp_solver"])
        ocp.solver_options.hessian_approx = solver_options.get(
            "hessian_approx", self.mpc_cfg.solver_options["hessian_approx"]
        )
        ocp.solver_options.integrator_type = solver_options.get(
            "integrator_type", self.mpc_cfg.solver_options["integrator_type"]
        )
        ocp.solver_options.nlp_solver_type = solver_options.get(
            "nlp_solver_type", self.mpc_cfg.solver_options["nlp_solver_type"]
        )
        ocp.solver_options.nlp_solver_max_iter = solver_options.get(
            "nlp_solver_max_iter", self.mpc_cfg.solver_options["nlp_solver_max_iter"]
        )
        ocp.solver_options.tol = solver_options.get("tol", self.mpc_cfg.solver_options["tol"])
        ocp.solver_options.tf = self.T * self.dt
        ocp.code_export_directory = self._code_export_dir
        return ocp

    def prepare_for_solve(
        self,
        solver,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray = None,
        u_sol_last: np.ndarray = None,
        x_sol_last: np.ndarray = None,
    ):
        """Prepare the MPC problem for solving at the current time step.

        Args:
            solver (AcadosOcp): Acados solver instance.
            x0: Current state.
            x_ref: Reference trajectory for the horizon.
            u_ref: Reference inputs for the horizon.
            u_sol_last: Previous solution for warm starting.
            x_sol_last: Previous solution for warm starting.
            which_params: List of parameter names to update.
        """
        # Set the initial state constraint
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        # Set the reference trajectory
        assert x_ref.shape == (self.T + 1, self.nx), (
            f"x_ref has shape {x_ref.shape}, but should be {(self.T + 1, self.nx)}"
        )
        if u_ref is None:
            u_ref = np.repeat(self.U_EQ[..., None], self.T, axis=-1).T
        assert u_ref.shape == (self.T, self.nu), f"u_ref has shape {u_ref.shape}, but should be {(self.T, self.nu)}"

        y_ref = np.hstack((x_ref[:-1, :], u_ref))  # shape (T, nx+nu)
        y_ref_e = x_ref[-1, :]  # shape (nx,)
        for t in range(self.T):
            solver.set(t, "yref", y_ref[t, :])
        solver.set(self.T, "yref", y_ref_e)

        # Warm start
        if u_sol_last is None:
            u_guess = u_ref
        else:
            u_guess = np.vstack([u_sol_last[1:], u_sol_last[-1]])
        if x_sol_last is None:
            x_guess = x_ref
        else:
            x_guess = np.vstack([x_sol_last[1:], x_sol_last[-1]])

        assert x_guess.shape == (self.T + 1, self.nx), f"Expected {(self.T + 1, self.nx)}, got {x_guess.shape}"
        assert u_guess.shape == (self.T, self.nu), f"Expected {(self.T, self.nu)}, got {u_guess.shape}"

        x_guess[0] = x0
        for k in range(self.T):
            solver.set(k, "x", x_guess[k])
            solver.set(k, "u", u_guess[k])
        solver.set(self.T, "x", x_guess[-1])

    def select_action(self, x0) -> np.ndarray:
        """Select action using MPC.

        Args:
            x0: Current state.

        Returns:
            u: Control action.
        """
        return self.step_control(self.ocp_solver, x0, self.registry, update_params=False)

    def step_control(
        self, solver, x0, registry: OcpRegistry, update_params: Union[list[str], bool] = False
    ) -> np.ndarray:
        """Solve the OCP and return the control action.

        Args:
            solver (AcadosOcpSolver): The solver instance.
            x0 (np.ndarray): Current state.
            registry (OcpRegistry): Parameter and constraint registry.
            update_params (list[str] or bool): Which parameters to update.

        Returns:
            np.ndarray: Control action.
        """
        x_ref = self.reference_trajectory()
        self.traj_step += 1

        self.prepare_for_solve(
            solver, x0, x_ref, u_ref=self.u_ref, x_sol_last=self.x_sol_last, u_sol_last=self.u_sol_last
        )
        # Update parameters if needed and set them in the solver
        registry.update_params(solver, which=update_params)

        status = solver.solve()
        if status in [0, 2]:  # 0: success, 2: success with warnings
            u = solver.get(0, "u")
            self.u_last = u  # Update last valid control
            self.x_sol_last = np.array([solver.get(i, "x") for i in range(self.T + 1)])
            self.u_sol_last = np.array([solver.get(i, "u") for i in range(self.T)])
        else:
            self.logger.warning(f"MPC solver failed with status {status}. Using LQR Gain.")
            u = self.lqr_gain @ (self.X_EQ - x0) + self.U_EQ
            u = np.clip(u, self.U_LOW, self.U_HIGH)
        return u

    def reference_trajectory(self) -> np.ndarray:
        """Construct reference states along the MPC horizon. Assumes periodicity.

        Returns:
            np.ndarray: Reference trajectory of shape (T+1, nx).
        """
        indices = np.arange(self.traj_step, self.traj_step + self.T + 1) % self.traj.shape[0]
        return self.traj[indices, :]

    def setup_prior_dynamics(self, cost_params: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set up prior dynamics for the MPC controller.

        Args:
            cost_params (dict | None): Cost weights containing 'Q' and 'R'. If None, uses self.mpc_cfg.cost_params.

        Returns:
            tuple: (A, B, lqr_gain) where
                A (np.ndarray): Discrete-time state matrix.
                B (np.ndarray): Discrete-time input matrix.
                lqr_gain (np.ndarray): Discrete-time LQR gain.

        """
        if cost_params is None:
            cost_params = self.mpc_cfg.cost_params
        dfdx_dfdu = self.symbolic_model.df_func(x=self.X_EQ, u=self.U_EQ)
        dfdx, dfdu = dfdx_dfdu["dfdx"].toarray(), dfdx_dfdu["dfdu"].toarray()
        A, B = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        Q = np.diag(cost_params.get("Q", self.mpc_cfg.cost_params["Q"]))
        R = np.diag(cost_params.get("R", self.mpc_cfg.cost_params["R"]))
        # ####################################################################
        # Task 3.2 Compute the (discrete) LQR gain for the terminal cost
        # Hint: take a look at the scipy.linalg functions
        # You need: A, B, Q, R
        ########################################################################
        lqr_gain = np.zeros((self.nu, self.nx)) # Compute this
        





        #######################################################################
        #                        END OF YOUR CODE
        #######################################################################
        assert lqr_gain.shape == (self.nu, self.nx), f"Expected {(self.nu, self.nx)}, got {lqr_gain.shape}"
        return A, B, lqr_gain

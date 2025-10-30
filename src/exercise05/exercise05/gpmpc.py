import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Tuple

import casadi as cs
import numpy as np
import scipy.stats
import torch
from acados_template import AcadosModel, AcadosOcpSolver  # type: ignore
from exercise05.gp import ResidualGP, ResidualGPConfig, ResidualGPTrainingConfig
from exercise05.mpc import MPC, MPCConfig, OcpRegistry
from exercise05.replay_buffer import ReplayBuffer, ReplayBufferConfig
from exercise05.utils import discretize_dynamics_rk4, recursive_update


@dataclass
class GPMPCConfig:
    mpc: MPCConfig = field(default_factory=MPCConfig)
    gp: ResidualGPConfig = field(default_factory=ResidualGPConfig)
    gp_train: ResidualGPTrainingConfig = field(default_factory=ResidualGPTrainingConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    gp_inputs: list = field(
        default_factory=lambda: [
            ["thrust"],
            ["phi", "dphi", "phi_cmd"],
            ["theta", "dtheta", "theta_cmd"],
        ]
    )
    gp_outputs: list = field(default_factory=lambda: ["acc", "dphi", "dtheta"])
    prob: float = 0.7  # For constraint tightening


class GPMPC(MPC):
    """Gaussian Process Model Predictive Controller (GP-MPC).

    Args:
        env: The environment object.
        prior_params (dict): Parameters for the prior model.
        output_dir: Directory to save outputs.
        cfg (dict, optional): Configuration dictionary. Defaults to {}.
        verbosity (str, optional): Logging verbosity level. Defaults to "info".
        device (str, optional): Device for PyTorch ('cpu' or 'cuda'). Defaults to "cpu".
        **kwargs: Additional keyword arguments.
    """

    gpmpc_cfg = GPMPCConfig()

    def __init__(
        self,
        env,
        output_dir,
        prior_params: dict = {},
        cfg={},
        verbosity: str = "info",
        device: str = "cpu",
        **kwargs: Any,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        log_level = getattr(logging, verbosity.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        self.verbosity = verbosity

        recursive_update(self.gpmpc_cfg, cfg)

        super().__init__(
            env=env,
            output_dir=output_dir,
            prior_params=prior_params,
            cfg=self.gpmpc_cfg.mpc,
            verbosity=verbosity,
            **kwargs,
        )

        self.device = device if torch.cuda.is_available() else "cpu"
        self.replay_buffer = ReplayBuffer(self.gpmpc_cfg.replay_buffer, verbosity=verbosity)

        self._X_train, self._Y_train = None, None  # Placeholder for initial data to setup GP
        # Inverse CDF for constraint tightening
        self.inverse_cdf = scipy.stats.norm.ppf(
            1 - (1 / self.nx - (self.gpmpc_cfg.prob + 1) / (2 * self.nx))
        )
        # Code creation variables
        self._gpmpc_json_file = str(self._output_dir / "gpmpc_ocp.json")
        self._gpmpc_json_sim_file = str(self._output_dir / "gpmpc_sim.json")
        self._gpmpc_code_export_dir = str(self._output_dir / "gpmpc_c_generated_code")

        self.acc_symbolic_fn = self.setup_symbolic_acceleration(self.prior_params)

        # Reset
        GPMPC.reset(self, full_reset=True)

    def setup_symbolic_acceleration(self, params: dict) -> cs.Function:
        """Set up symbolic mapping from normalized thrust command to actual thrust.

        Args:
            params (dict): Parameters with keys 'a' and 'b' for linear mapping.
        """
        T = cs.MX.sym("T_c")
        T_mapping = params["a"] * T + params["b"]
        return cs.Function("T_mapping", [T], [T_mapping])

    def reset(self, full_reset: bool = False):
        """Reset the GP-MPC controller.

        Args:
            full_reset (bool): If True, redeploy GP and GP-MPC. If False, only reset the controller state.
        """
        super().reset(full_reset=full_reset)
        if full_reset:
            self.gpmpc_registry = OcpRegistry(self.T)
            self.replay_buffer.clear()
            self._reset_gp()
            self._reset_gpmpc()
        else:
            if self.gpmpc_solver is not None:
                self.gpmpc_solver.reset()

    def _reset_gp(self):
        self._gp_initialized = False
        self.gp: ResidualGP = None
        self.setup_gp(override=True)

    def _reset_gpmpc(self):
        self._gpmpc_initialized = False
        self.gpmpc_ocp = None
        self.gpmpc_solver = None
        self.setup_gpmpc(override=True)

    def setup_gp(self, X_init=None, Y_init=None, override: bool = False):
        if X_init is None or Y_init is None:
            self.logger.debug("Skipping GP setup; no initial data provided.")
            return
        if self._gp_initialized and not override:
            return

        self.gp: ResidualGP = ResidualGP(
            X_list=X_init,
            Y_list=Y_init,
            gp_cfg=self.gpmpc_cfg.gp,
            train_cfg=self.gpmpc_cfg.gp_train,
            verbosity=self.verbosity,
        )
        self._X_train, self._Y_train = X_init, Y_init
        self._gp_initialized = True
        self.logger.info("GP initialized.")

    def setup_gpmpc(self, override: bool = False):
        if not self._gp_initialized:
            return
        if self._gpmpc_initialized and not override:
            return

        # Copy & extend the acados model with GP residual dynamics
        acados_model: AcadosModel = copy.deepcopy(self.acados_model)
        acados_model = self._add_residual_dynamics_to_acados_model(acados_model)

        ocp = self.setup_acados_ocp(acados_model, cfg=self.mpc_cfg)
        ocp.model.name = "gpmpc"
        ocp.solver_options.integrator_type = "DISCRETE"  # Enforce discrete dynamics
        ocp.code_export_directory = self._gpmpc_code_export_dir

        self._modify_constraints_for_tightening(acados_model)

        self.gpmpc_registry.build_parameters(ocp)
        self.gpmpc_registry.build_constraints(ocp)

        # Create the solver
        self.gpmpc_model = acados_model
        self.gpmpc_ocp = ocp
        self.gpmpc_solver = AcadosOcpSolver(ocp, json_file=self._gpmpc_json_file, verbose=False)
        self._gpmpc_initialized = True
        self.logger.debug("GPMPC initialized.")

    def _add_residual_dynamics_to_acados_model(self, acados_model: AcadosModel) -> AcadosModel:
        """Modify the AcadosModel to include GP dynamics. Hardcoded for sparse parameterization."""
        x = acados_model.x
        u = acados_model.u
        z = cs.vertcat(x, u)
        state_mapping = self.mpc_cfg.state_mapping
        action_mapping = self.mpc_cfg.action_mapping
        gp_inputs = self.gpmpc_cfg.gp_inputs
        gp_outputs = self.gpmpc_cfg.gp_outputs
        # Extract the individual gp inputs
        idx_lists = []
        for input_vars in gp_inputs:
            idx_list = []
            for var in input_vars:
                if var in state_mapping:
                    idx_list.append(state_mapping[var])
                elif var in action_mapping:
                    idx_list.append(self.nx + action_mapping[var])
                else:
                    raise ValueError(f"Variable '{var}' not found in state or action mapping.")
            idx_lists.append(idx_list)
        Z = [z[idx_list] for idx_list in idx_lists]
        # Get the residual dynamics expression and required parameters from the GP
        sym_residual_exprs, sym_gp_params_list = self.gp.get_sparse_parameterization(Z=Z)

        T_pred = sym_residual_exprs[gp_outputs.index("acc")]
        phi_idx, theta_idx = state_mapping["phi"], state_mapping["theta"]
        ax_sym = T_pred * (cs.cos(acados_model.x[phi_idx]) * cs.sin(acados_model.x[theta_idx]))
        ay_sym = T_pred * (-cs.sin(acados_model.x[phi_idx]))
        az_sym = T_pred * (cs.cos(acados_model.x[phi_idx]) * cs.cos(acados_model.x[theta_idx]))

        # Build full residual vector safely (no in-place MX mutation)
        residual_list = [cs.MX.zeros(1) for _ in range(self.nx)]
        residual_list[state_mapping["dx"]] = ax_sym
        residual_list[state_mapping["dy"]] = ay_sym
        residual_list[state_mapping["dz"]] = az_sym
        residual_list[state_mapping["dphi"]] = sym_residual_exprs[gp_outputs.index("dphi")]
        residual_list[state_mapping["dtheta"]] = sym_residual_exprs[gp_outputs.index("dtheta")]
        sym_residual_expr = cs.vertcat(*residual_list)

        # Flatten GP parameter symbols
        sym_gp_params = cs.vertcat(*sym_gp_params_list)

        # Register GP parameters
        self.gpmpc_registry.add_param(
            "gp_params",
            size=int(sym_gp_params.shape[0]),
            update_fcn=self.gp.update_residual_parameters,
            overwrite=True,
            sym=sym_gp_params,
        )

        # Compose new discrete dynamics: x_{k+1} = f_nom(x,u) + residual(x,u)
        f_cont_nom = self.symbolic_model.fc_func(x, u)
        f_cont = f_cont_nom + sym_residual_expr
        f_cont_fun = cs.Function("f_cont_fun", [x, u, sym_gp_params], [f_cont])
        acados_model.disc_dyn_expr = discretize_dynamics_rk4(
            f_cont_fun, self.dt, x, u, sym_gp_params, as_function=False
        )
        self.f_disc_nom = cs.Function("f_disc_nom", [x, u], [self.acados_model.disc_dyn_expr])
        self.f_disc_gp = cs.Function(
            "f_disc_gp", [x, u, sym_gp_params], [acados_model.disc_dyn_expr]
        )

        # Remove continuous forms (since integrator_type=DISCRETE)
        acados_model.f_expl_expr = None
        acados_model.f_impl_expr = None
        return acados_model

    def update_tightening_parameters(self, max_tightening: float = 0.3) -> np.ndarray:
        """Propagate state covariance under GP residual (process) noise and compute box constraint tightening margins. Partially hardcoded for sparse parameterization.

        Assumes positive tightening parameters are added to the left-hand side of the constraints, e.g., Ax <= b  -> Ax <= b - margin.

        Args:
            max_tightening (float): Maximum allowed tightening margin as a fraction of the constraint range.

        Returns:
            np.ndarray of shape (2*nx + 2*nu, T+1):
                [ state_margins (2*nx, T+1);
                  input_margins (2*nu, T+1) ]
        """
        nx, nu, T = self.nx, self.nu, self.T
        margins_state = np.zeros((2 * nx, T + 1))
        margins_input = np.zeros((2 * nu, T + 1))

        # If probability tightening disabled or missing trajectory, return zeros
        if (
            self.gpmpc_cfg.prob <= 0.0
            or self.x_sol_last is None
            or self.u_sol_last is None
            or not self._gp_initialized
        ):
            return np.concatenate([margins_state, margins_input], axis=0)

        state_mapping = self.mpc_cfg.state_mapping
        action_mapping = self.mpc_cfg.action_mapping
        # Construct linearization matrices and residual injection matrix
        A_k = self.prior_A  # (nx, nx)
        B_k = self.prior_B  # (nx, nu)
        # Residual mapping matrix Bd: Assumes residual outputs order = ["acc", "dphi", "dtheta"]
        Bd = np.zeros((nx, len(self.gpmpc_cfg.gp_outputs)))  # (nx, n_res)
        # Acceleration residual affects velocity derivatives
        idx_vx, idx_vy, idx_vz = state_mapping["dx"], state_mapping["dy"], state_mapping["dz"]
        idx_dphi, idx_dtheta = state_mapping["dphi"], state_mapping["dtheta"]
        # Simple isotropic split of scalar thrust residual into xyz:
        Bd[idx_vx, 0] = 1.0 / np.sqrt(3)
        Bd[idx_vy, 0] = 1.0 / np.sqrt(3)
        Bd[idx_vz, 0] = 1.0 / np.sqrt(3)
        Bd[idx_dphi, 1] = 1.0
        Bd[idx_dtheta, 2] = 1.0

        # Extract planned trajectories
        Xh = self.x_sol_last  # (T+1, nx)
        Uh = self.u_sol_last  # (T, nu)

        # Build grouped inputs for each GP
        grouped_inputs = []
        for group in self.gpmpc_cfg.gp_inputs:
            cols = []
            for var in group:
                if var in state_mapping:
                    cols.append(Xh[:-1, state_mapping[var]])
                elif var in action_mapping:
                    cols.append(Uh[:, action_mapping[var]])
                else:
                    raise KeyError(f"Variable {var} not found in mappings.")
            grouped_inputs.append(np.stack(cols, axis=1))  # (T, group_dim)

        # Query predictive variances from each single-output GP
        gp_vars = []
        gp_means = []
        for gp_model, Z in zip(self.gp._gps, grouped_inputs):
            mean, var = gp_model.predict(torch.tensor(Z, dtype=torch.float32))
            gp_vars.append(var.numpy())  # shape (T,)
            gp_means.append(mean.numpy())

        # Ensure ordering matches gp_outputs
        # gp_outputs default: ["acc", "dphi", "dtheta"]
        output_order = self.gpmpc_cfg.gp_outputs
        var_dict = {name: v for name, v in zip(output_order, gp_vars)}
        # Build (T, n_res)
        residual_var = np.vstack([var_dict["acc"], var_dict["dphi"], var_dict["dtheta"]]).T
        residual_var = np.maximum(residual_var, 1e-10)
        # ####################################################################
        # Task 3.3 Compute covariance propagation and constraint tightening margins.
        # For each k in 0,...,T-1:
        # - Unpack residual_var at time k to get process noise covariance Sigma_d
        # - Compute input covariance Sigma_u via local feedback δu ≈ K δx
        # - Compute state-input covariance Sigma_xu
        # - Compute margins at time k using z and the diagonal of Sigma_x and Sigma_u
        # - Propagate state covariance to next time step using the linearized dynamics
        # Compute terminal margins at time T
        ########################################################################
        # Show variables
        K = self.lqr_gain  # (nu, nx)
        z = self.inverse_cdf
        A_k = A_k
        B_k = B_k
        Bd = Bd
        Sigma_d, Sigma_u, Sigma_xu, Sigma_ux = (
            None,
            None,
            None,
            None,
        )  # Compute those in each iteration
        Sigma_x = np.zeros((nx, nx))  # Initialize state covariance
        


































        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        assert margins_state.shape == (2 * nx, T + 1)
        assert margins_input.shape == (2 * nu, T + 1)

        # Clip to maximum allowed tightening
        state_range = self.X_HIGH - self.X_LOW
        input_range = self.U_HIGH - self.U_LOW
        for i in range(2 * nx):
            margins_state[i, :] = np.minimum(
                margins_state[i, :], max_tightening * state_range[i % nx]
            )
        for i in range(2 * nu):
            margins_input[i, :] = np.minimum(
                margins_input[i, :], max_tightening * input_range[i % nu]
            )

        return np.concatenate([margins_state, margins_input], axis=0)

    def _modify_constraints_for_tightening(self, acados_model: AcadosModel):
        """Define tightened state and input constraints by adding the tightening parameters on the left side, e.g., Ax <= b  -> Ax <= b - margin."""
        tightening_params = self.gpmpc_registry.add_param(
            name="tightening",
            size=2 * self.nx + 2 * self.nu,
            update_fcn=self.update_tightening_parameters,
        )

        state_constr, input_constr = self.create_tightened_constraints(
            tightening_params=tightening_params, x=acados_model.x, u=acados_model.u
        )

        self.gpmpc_registry.add_constraint(
            "state", state_constr, ["path", "terminal"], allow_slack=False
        )
        self.gpmpc_registry.add_constraint(
            "input", input_constr, ["initial", "path"], allow_slack=False
        )

    def create_tightened_constraints(
        self, tightening_params: cs.MX, x: cs.MX, u: cs.MX
    ) -> Tuple[cs.MX, cs.MX]:
        """Create tightened state and input constraint expressions for given tightening parameters.

        Args:
            tightening_params (cs.MX): Tightening parameters of shape (2*nx + 2*nu,).
            x (cs.MX): State variable of shape (nx,).
            u (cs.MX): Input variable of shape (nu,).

        Returns:
            Tuple[cs.MX, cs.MX]: State and input constraint expressions with shape (2*nx,) and (2*nu,), respectively.
        """
        #####################################################################
        # Task 3.1 Create symbolic expressions for tightened box constraints in the form
        # Ax <= b  -> Ax <= b - margin
        # where margin is a positive tightening parameter added to the left-hand side.
        #####################################################################
        nx = x.shape[0]
        nu = u.shape[0]
        state_tightening_params = tightening_params[: 2 * nx]
        input_tightening_params = tightening_params[2 * nx :]
        X_low = self.X_LOW[:nx]
        X_high = self.X_HIGH[:nx]
        U_low = self.U_LOW[:nu]
        U_high = self.U_HIGH[:nu]
        state_constr: cs.MX = None  # compute this
        input_constr: cs.MX = None  # compute this

        # ####################################################################
        









        ######################################################################
        #                           END OF YOUR CODE
        ######################################################################
        return state_constr, input_constr

    def select_action(self, x0):
        if not self._gpmpc_initialized:
            self.ctrl_type = "MPC"
            return super().select_action(x0)
        else:
            self.ctrl_type = "GPMPC"
            u = self.step_control(
                solver=self.gpmpc_solver,
                x0=x0,
                registry=self.gpmpc_registry,
                update_params=["tightening"],
            )
        self.logger.debug(f"Used controller: {self.ctrl_type}")
        return u

    def _preprocess_data(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray):
        """Process trajectory data for the Gaussian Processes.

        Note: This is hardcoded for low-dimensional ResidualGP for the attitude model.

        Args:
            x (NDArray): State sequence of shape (N, nx).
            u (NDArray): Action sequence of shape (N, nu).
            x_next (NDArray): Next state sequence of shape (N, nx).

        Returns:
            Tuple[NDArray, NDArray]: Inputs and targets for GP training, shapes (N, 7) and (N, 3).
        """
        # Unpack mappings
        state_mapping = self.mpc_cfg.state_mapping
        action_mapping = self.mpc_cfg.action_mapping
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        g = 9.81  # Gravity constant
        dt = self.dt

        x_dot = (x_next - x) / dt  # Approximate via numerical differentiation
        thrust_idx = action_mapping["thrust"]
        thrust_cmd = u[:, thrust_idx]

        # Faster than broadcasted version of np.linalg.norm
        dx_idx, dy_idx, dz_idx = state_mapping["dx"], state_mapping["dy"], state_mapping["dz"]
        acc = np.sqrt(x_dot[:, dx_idx] ** 2 + x_dot[:, dy_idx] ** 2 + (x_dot[:, dz_idx] + g) ** 2)
        acc_prior = self.acc_symbolic_fn(thrust_cmd).full().flatten()
        acc_target = acc - acc_prior
        acc_input = thrust_cmd.reshape(-1, 1)

        phi_idx, dphi_idx, phi_cmd_idx = (
            state_mapping["phi"],
            state_mapping["dphi"],
            action_mapping["phi_cmd"],
        )
        dphi_meas = x_dot[:, phi_idx]
        dphi_prior = self.symbolic_model.fc_func(x=x.T, u=u.T)["f"].toarray()[phi_idx, :]
        phi_target = dphi_meas - dphi_prior
        phi_input = np.vstack((x[:, phi_idx], x[:, dphi_idx], u[:, phi_cmd_idx])).T

        theta_idx, dtheta_idx, dtheta_cmd_idx = (
            state_mapping["theta"],
            state_mapping["dtheta"],
            action_mapping["theta_cmd"],
        )
        dtheta_meas = x_dot[:, theta_idx]
        dtheta_prior = self.symbolic_model.fc_func(x=x.T, u=u.T)["f"].toarray()[theta_idx, :]
        theta_target = dtheta_meas - dtheta_prior
        theta_input = np.vstack((x[:, theta_idx], x[:, dtheta_idx], u[:, dtheta_cmd_idx])).T

        train_input = np.concatenate([acc_input, phi_input, theta_input], axis=-1)
        train_output = np.vstack((acc_target, phi_target, theta_target)).T
        return train_input, train_output

    def add_data(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray):
        """Add new data for residual training to the replay buffer."""
        if x.ndim == 1:  # single sample fallback
            x = x[None, :]
            u = u[None, :]
            x_next = x_next[None, :]
        x_train, y_train = self._preprocess_data(x, u, x_next)
        # Detach to CPU for buffer (assuming buffer expects numpy)
        x_train = torch.tensor(x_train, dtype=torch.float32).cpu()
        y_train = torch.tensor(y_train, dtype=torch.float32).cpu()
        self.replay_buffer.add(x_train, y_train)

    def fit(self):
        X, Y = self.replay_buffer.get_dataset()
        if X is None or X.shape[0] == 0:
            self.logger.debug("No data in replay buffer; skipping GP train.")
            return
        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32)
        # Split data for each GP (Harcoded see _preprocess_data for details)
        X_list = [X_t[:, :1], X_t[:, 1:4], X_t[:, 4:]]
        Y_list = [Y_t[:, :1], Y_t[:, 1:2], Y_t[:, 2:]]
        if not self._gp_initialized:
            # Use initial feature data for inducing point selection
            self.setup_gp(X_list, Y_list, override=True)

        self.gp.fit(X_list, Y_list)
        # After first training, build solver if not built
        if not self._gpmpc_initialized:
            self.setup_gpmpc(override=True)

        self.gpmpc_registry.update_params(self.gpmpc_solver, which=["gp_params"])

    def evaluate_dynamics_quality(
        self, test_x: np.ndarray, test_u: np.ndarray, test_x_next: np.ndarray, plot: bool = True
    ):
        """Evaluate and compare the prediction quality of the nominal model and GP-augmented model.

        Args:
            test_x: (N, nx) Test states.
            test_u: (N, nu) Test actions.
            test_x_next: (N, nx) True next states.
            plot: Whether to plot results.
        """
        assert self._gp_initialized, "GP not initialized."
        assert self._gpmpc_initialized, "GP-MPC not initialized."
        assert test_x.shape[0] == test_u.shape[0] == test_x_next.shape[0], (
            "Mismatched test data lengths."
        )
        # Prepare inputs for CasADi functions (shape: (nx, N), (nu, N))
        x_in = test_x.T
        u_in = test_u.T
        N = test_x.shape[0]

        # Get predictions
        f_nom_map = self.f_disc_nom.map(N)
        x_pred_nom = f_nom_map(x_in, u_in).full().T  # (N, nx)
        f_gp_map = self.f_disc_gp.map(N)
        gp_params = self.gp.update_residual_parameters()  # Get last trained parameters
        x_pred_gp = f_gp_map(x_in, u_in, gp_params).full().T  # (N, nx)

        # Compute errors
        err_nom = np.linalg.norm(x_pred_nom - test_x_next, axis=1)
        err_gp = np.linalg.norm(x_pred_gp - test_x_next, axis=1)
        print(f"Nominal RMSE: {np.sqrt(np.mean(err_nom**2)):.4f}")
        print(f"GP-augmented RMSE: {np.sqrt(np.mean(err_gp**2)):.4f}")

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4))
            plt.plot(err_nom, label="Nominal error")
            plt.plot(err_gp, label="GP-augmented error")
            plt.xlabel("Sample")
            plt.ylabel("Prediction error (L2 norm)")
            plt.legend()
            plt.title("Dynamics Prediction Error")
            plt.show()

            plt.figure()
            plt.hist(err_nom, bins=30, alpha=0.5, label="Nominal")
            plt.hist(err_gp, bins=30, alpha=0.5, label="GP-augmented")
            plt.xlabel("Prediction error (L2 norm)")
            plt.ylabel("Count")
            plt.legend()
            plt.title("Prediction Error Histogram")
            plt.show()

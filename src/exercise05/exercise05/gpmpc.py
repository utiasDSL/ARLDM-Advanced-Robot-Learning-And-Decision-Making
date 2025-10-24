"""GP-MPC controller using Acados for optimization."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import casadi as cs
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from exercise05.gp import (
    GaussianProcess,
    covSE_vectorized,
    fit_gp,
    gpytorch_predict2casadi,
    infer_gp,
)
from exercise05.mpc import MPC
from exercise05.utils import (
    discretize_dynamics_rk4,
    discretize_linear_system,
    remove_path,
)
from gpytorch.settings import fast_pred_samples, fast_pred_var
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class GPMPC:
    """Implements a GP-MPC controller with Acados optimization."""

    U_EQ: NDArray = np.array([0.3234, 0, 0, 0])

    def __init__(
        self,
        symbolic_model,
        traj: NDArray,
        prior_params: dict,
        horizon: int,
        q_mpc: List[float],
        r_mpc: List[float],
        sparse_gp: bool = False,
        prob: float = 0.955,
        max_gp_samples: int = 30,
        seed: int = 1337,
        device: str = "cpu",
        output_dir: Path = Path("results/temp"),
        action_space: Optional[dict] = None,
    ):
        """Initialize the GP-MPC controller.

        Args:
            symbolic_model: The symbolic model for the system.
            traj (NDArray): Reference trajectory.
            prior_params (dict): Prior parameters for the model.
            horizon (int): MPC horizon.
            q_mpc (List[float]): State cost weights.
            r_mpc (List[float]): Input cost weights.
            sparse_gp (bool): Whether to use sparse GP.
            prob (float): Probability for constraint tightening.
            max_gp_samples (int): Maximum GP samples for sparse GP.
            seed (int): Random seed.
            device (str): Torch device.
            output_dir (Path): Output directory.
            action_space (Optional[dict]): Action space constraints (usually the action space of the environment).
        """
        self.sparse = sparse_gp
        self.output_dir = output_dir
        self.json_file = str(self.output_dir / "gpmpc_acados_ocp_solver.json")
        if "cuda" in device and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Model parameters
        self.model = symbolic_model
        if prior_params is None or any(k not in prior_params for k in ("a", "b")):
            raise ValueError("prior_params must contain keys 'a' and 'b'")
        self.acc_symbolic_fn = self.setup_symbolic_acceleration(prior_params)
        self.dt = self.model.dt
        self.T = horizon
        assert len(q_mpc) == self.model.nx and len(r_mpc) == self.model.nu
        self.Q = np.diag(q_mpc)
        self.R = np.diag(r_mpc)

        # Setup references.
        self.traj = traj
        self.ref_action = np.repeat(self.U_EQ[..., None], self.T, axis=1)
        self.traj_step = 0
        self.np_random = np.random.default_rng(seed)

        # GP and training parameters.
        self.gp_idx = [[0], [1, 2, 3], [4, 5, 6]]  # State input indices for each GP
        self.gaussian_process = None
        self._requires_recompile = False
        self.gp_state_input = None
        self.inverse_cdf = scipy.stats.norm.ppf(
            1 - (1 / self.model.nx - (prob + 1) / (2 * self.model.nx))
        )
        self.max_gp_samples = max_gp_samples

        uncertain_dim = [1, 3, 5, 9, 10]  # dx, dy, dz, dphi, dtheta
        self.Bd = np.eye(self.model.nx)[:, uncertain_dim]

        # MPC params
        prior_ctrl = MPC(
            symbolic_model,
            traj=traj,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            output_dir=output_dir,
            action_space=action_space,
        )
        self.action_space = action_space
        self.prior_ctrl = prior_ctrl  # Required for selecting actions with the prior dynamics only
        x_eq, u_eq = np.zeros(self.model.nx), self.U_EQ
        dfdx_dfdu = prior_ctrl.model.df_func(x=x_eq, u=u_eq)
        prior_dfdx, prior_dfdu = dfdx_dfdu["dfdx"].toarray(), dfdx_dfdu["dfdu"].toarray()
        self.discrete_dfdx, self.discrete_dfdu, self.lqr_gain = self.setup_prior_dynamics(
            prior_dfdx, prior_dfdu, self.Q, self.R, self.dt
        )
        self.prior_dynamics: cs.Function = prior_ctrl.model.fc_func
        assert isinstance(self.prior_dynamics, cs.Function)

        self.acados_solver: Optional[AcadosOcpSolver] = (
            None  # Gets compiled in reset() if GP has been trained
        )
        self.x_prev: Optional[NDArray] = None
        self.u_prev: Optional[NDArray] = None

    def reset(self):
        """Reset the controller before running."""
        self.traj_step = 0
        if self._requires_recompile:  # GPs have changed, Acados must be recompiled
            assert self.gaussian_process is not None, "GP must be trained before reinitializing"
            n_samples = self.gaussian_process[0].train_targets.shape[0]
            if self.sparse:
                n_samples = min(n_samples, self.max_gp_samples)
            # Reinitialize Acados model and solver -> leads to recompilation of the Acados solver
            acados_model = self.setup_acados_model(n_samples)
            ocp = self.setup_acados_optimizer(acados_model, n_samples)
            remove_path(ocp.code_export_directory)  # Remove old code directory if it exists
            remove_path(self.json_file)  # Remove old json file if it exists
            self.acados_solver = AcadosOcpSolver(ocp, json_file=self.json_file, verbose=False)
            self._requires_recompile = False
        # Previously solved states & inputs
        self.x_prev = None
        self.u_prev = None

    def preprocess_data(self, x: NDArray, u: NDArray, x_next: NDArray) -> tuple[NDArray, NDArray]:
        """Convert trajectory data for GP trianing.

        Args:
            x (NDArray): State sequence of shape (N, nx).
            u (NDArray): Action sequence of shape (N, nu).
            x_next (NDArray): Next state sequence of shape (N, nx).

        Returns:
            Tuple[NDArray, NDArray]: Inputs and targets for GP training, shapes (N, nx+nu) and (N, nx).
        """
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        g = 9.81
        dt = self.dt
        thrust_cmd = u[:, 0]
        x_dot = (x_next - x) / dt  # Approximate via numerical differentiation

        # Faster than broadcasted version of np.linalg.norm
        acc = np.sqrt(x_dot[:, 1] ** 2 + x_dot[:, 3] ** 2 + (x_dot[:, 5] + g) ** 2)
        acc_prior = self.acc_symbolic_fn(thrust_cmd).full().flatten()
        acc_target = acc - acc_prior
        acc_input = thrust_cmd.reshape(-1, 1)

        idx_phi, idx_d_phi, idx_phi_cmd = 6, 9, 1
        phi = x_dot[:, idx_phi]
        phi_prior = self.prior_dynamics(x=x.T, u=u.T)["f"].toarray()[idx_phi, :]
        phi_target = phi - phi_prior
        phi_input = np.vstack((x[:, idx_phi], x[:, idx_d_phi], u[:, idx_phi_cmd])).T

        idx_theta, idx_d_theta, idx_theta_cmd = 7, 10, 2
        theta = x_dot[:, idx_theta]
        theta_prior = self.prior_dynamics(x=x.T, u=u.T)["f"].toarray()[idx_theta, :]
        theta_target = theta - theta_prior
        theta_input = np.vstack((x[:, idx_theta], x[:, idx_d_theta], u[:, idx_theta_cmd])).T

        train_input = np.concatenate([acc_input, phi_input, theta_input], axis=-1)
        train_output = np.vstack((acc_target, phi_target, theta_target)).T
        return train_input, train_output

    def train_gp(self, x: NDArray, y: NDArray, lr: float, iterations: int, val_split: float = 0.2):
        """Fit the GPs to the training data.

        Args:
            x (NDArray): Training inputs, shape (N, D).
            y (NDArray): Training targets, shape (N, O).
            lr (float): Learning rate for GP optimization.
            iterations (int): Number of optimization iterations.
            val_split (float): Fraction of data to use for validation dataset (0-1).
        """
        assert val_split >= 0 and val_split <= 1, "Expected val_split to be in [0, 1]"
        x_train = torch.tensor(x, device=self.device)
        y_train = torch.tensor(y, device=self.device)
        n = x_train.shape[0]
        n_val = int(n * val_split)
        if n_val > 0:
            # Shuffle the training data and split into training and validation sets
            idx = torch.randperm(n)
            val_x, val_y = x_train[idx[:n_val]], y_train[idx[:n_val]]
            x_train, y_train = x_train[idx[n_val:]], y_train[idx[n_val:]]
        else:
            val_x, val_y = None, None
        ########################################################################
        # Task 2.2                                                             
        # TODO:                                                                
        # 1. For each group of input indices in "self.gp_idx":                 
        #    a. Extract the relevant columns from the 'x_train'                
        #       for this GP.                                                   
        #    b. Extract the corresponding target output from `y`.              
        #    c. Create a GaussianProcess object for this input-output pair.    
        #    d. Fit the GP using the provided learning rate, number of         
        #       iterations, optional (extracted) validation data,              
        #       and imported `fit_gp` function,  (Task 1.2)                    
        # 2. Store all trained GaussianProcess objects in                      
        #    `self.gaussian_process` as a list.                                
        ########################################################################
        










        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        self._requires_recompile = True

    def setup_acados_model(self, n_samples: int) -> AcadosModel:
        """Setup the Acados model for MPC.

        Args:
            n_samples (int): Number of samples for GP approximation.

        Returns:
            AcadosModel: Configured Acados model.
        """
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym

        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point
        nx = self.model.nx
        idx_T, idx_R, idx_P = [0 + nx], [6, 9, 1 + nx], [7, 10, 2 + nx]

        if self.sparse:
            K_zs_T_cs, K_zs_R_cs, K_zs_P_cs = self.sparse_gp_kernels_cs(n_samples)

            x_sparse = cs.MX.sym("x_sparse", n_samples, 7)
            posterior_mean = cs.MX.sym("posterior_mean", 3, n_samples)
            # Acados supports only 1D parameters
            flat_x_sparse = cs.reshape(x_sparse, -1, 1)
            flat_posterior_mean = cs.reshape(posterior_mean, -1, 1)
            acados_model.p = cs.vertcat(flat_x_sparse, flat_posterior_mean)

            T_pred = cs.sum2(K_zs_T_cs(z1=z[idx_T], z2=x_sparse)["K"] * posterior_mean[0, :])
            R_pred = cs.sum2(K_zs_R_cs(z1=z[idx_R], z2=x_sparse)["K"] * posterior_mean[1, :])
            P_pred = cs.sum2(K_zs_P_cs(z1=z[idx_P], z2=x_sparse)["K"] * posterior_mean[2, :])
        else:
            T_pred = gpytorch_predict2casadi(self.gaussian_process[0])(z=z[idx_T])["mean"]
            R_pred = gpytorch_predict2casadi(self.gaussian_process[1])(z=z[idx_R])["mean"]
            P_pred = gpytorch_predict2casadi(self.gaussian_process[2])(z=z[idx_P])["mean"]

        idx_phi, idx_theta = 6, 7
        ax_sym = T_pred * (cs.cos(acados_model.x[idx_phi]) * cs.sin(acados_model.x[idx_theta]))
        ay_sym = T_pred * (-cs.sin(acados_model.x[idx_phi]))
        az_sym = T_pred * (cs.cos(acados_model.x[idx_phi]) * cs.cos(acados_model.x[idx_theta]))
        res_dyn = cs.vertcat(0, ax_sym, 0, ay_sym, 0, az_sym, 0, 0, 0, R_pred, P_pred, 0)

        f_cont = self.prior_dynamics(x=acados_model.x, u=acados_model.u)["f"] + res_dyn
        f_cont_fn = cs.Function(
            "f_cont_fn", [acados_model.x, acados_model.u, acados_model.p], [f_cont]
        )

        # Use rk4 to discretize the continuous dynamics
        acados_model.disc_dyn_expr = discretize_dynamics_rk4(
            f_cont_fn, self.dt, acados_model.x, acados_model.u, p=acados_model.p, as_function=False
        )

        # It is critical to change the name of the model for different number of samples. Acados will silently reuse parts of a previous model if the names are the same. This leads to shape mismatch errors in the constraints, which grow with the number of samples. Hence, we add a postfix to the model name that changes with the number of samples used to construct the model and avoid the reuse of params. See: https://github.com/acados/acados/issues/905
        acados_model.name = "gpmpc" + str(n_samples)
        acados_model.t_label = "time"

        return acados_model

    def setup_acados_optimizer(self, acados_model: AcadosModel, n_samples: int) -> AcadosOcp:
        """Setup the Acados optimizer.

        Args:
            acados_model (AcadosModel): The Acados model.
            n_samples (int): Number of samples for GP approximation.

        Returns:
            AcadosOcp: Configured Acados OCP.
        """
        ocp = AcadosOcp()
        ocp.model = acados_model
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # Configure costs
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx : (nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Constraints
        s_low = np.array([-2, -15, -2, -15, -0.05, -15, -1.5, -1.5, -10, -8.5, -8.5, -10])
        s_high = np.array([2, 15, 2, 15, 2, 15, 1.5, 1.5, 10, 8.5, 8.5, 10])
        state_cnstr = self.setup_constraints(ocp.model.x, s_low, s_high)
        if self.action_space is not None:
            u_low = self.action_space["low"]
            u_high = self.action_space["high"]
        else:
            u_low = np.array([0.12, -0.43, -0.43, -0.43])
            u_high = np.array([0.59, 0.43, 0.43, 0.43])
        input_cnstr = self.setup_constraints(ocp.model.u, u_low, u_high)
        ocp = self.setup_acados_constraints(ocp, state_cnstr, input_cnstr)

        # Initialize with placeholder zero values
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
        ocp.constraints.x0 = np.zeros((nx))

        # Set up solver options
        ocp.solver_options.N_horizon = self.T  # prediction horizon
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 25
        ocp.solver_options.tf = self.T * self.dt  # prediction duration
        ocp.code_export_directory = str(self.output_dir / "gpmpc_c_generated_code")

        if self.sparse:
            posterior_mean, train_inputs = self.precompute_sparse_posterior_mean(n_samples)
        else:
            posterior_mean, train_inputs = self.precompute_posterior_mean()
        # Casadi use column major order, while np uses row major order by default
        train_inputs = train_inputs.reshape(-1, 1, order="F")
        posterior_mean = posterior_mean.reshape(-1, 1, order="F")
        dyn_value = np.concatenate((train_inputs, posterior_mean), axis=0)
        self.gp_state_input = np.tile(dyn_value, (1, self.T + 1))
        return ocp

    def setup_acados_constraints(
        self, ocp: AcadosOcp, state_cnstr: cs.MX, input_cnstr: cs.MX
    ) -> AcadosOcp:
        """Setup state and input constraints for Acados OCP.

        Args:
            ocp (AcadosOcp): The Acados OCP.
            state_cnstr (cs.MX): State constraints.
            input_cnstr (cs.MX): Input constraints.

        Returns:
            AcadosOcp: OCP with constraints applied.
        """
        initial_cnstr = cs.vertcat(state_cnstr, input_cnstr)
        cnstr = cs.vertcat(state_cnstr, input_cnstr)
        terminal_cnstr = cs.vertcat(state_cnstr)  # Terminal constraints are only state constraints
        state_tighten_param = cs.MX.sym("state_tighten_param", *state_cnstr.shape)
        input_tighten_param = cs.MX.sym("input_tighten_param", *input_cnstr.shape)
        state_input_tighten_param = cs.vertcat(state_tighten_param, input_tighten_param)

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0 = initial_cnstr - state_input_tighten_param
        ocp.model.con_h_expr = cnstr - state_input_tighten_param
        ocp.model.con_h_expr_e = terminal_cnstr - state_tighten_param
        ocp.dims.nh_0 = initial_cnstr.shape[0]
        ocp.dims.nh = cnstr.shape[0]
        ocp.dims.nh_e = terminal_cnstr.shape[0]

        # All constraints are defined as g(x, u) <= tol. Acados requires the constraints to be
        # defined as lb <= g(x, u) <= ub. Thus, a large negative number (-1e8) is used as the lower
        # bound to ensure that the constraints are not active. np.prod makes sure all ub and lb are
        # 1D numpy arrays
        # See: https://github.com/acados/acados/issues/650
        # See: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche
        ocp.constraints.uh_0 = -1e-8 * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.lh_0 = -1e8 * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.uh = -1e-8 * np.ones(np.prod(cnstr.shape))
        ocp.constraints.lh = -1e8 * np.ones(np.prod(cnstr.shape))
        ocp.constraints.uh_e = -1e-8 * np.ones(np.prod(terminal_cnstr.shape))
        ocp.constraints.lh_e = -1e8 * np.ones(np.prod(terminal_cnstr.shape))

        # Pass the tightening variables to the ocp object as parameters
        tighten_param = cs.vertcat(state_tighten_param, input_tighten_param)
        ocp.model.p = cs.vertcat(ocp.model.p, tighten_param) if self.sparse else tighten_param

        return ocp

    def setup_symbolic_acceleration(self, params: dict) -> cs.Function:
        T = cs.MX.sym("T_c")
        T_mapping = params["a"] * T + params["b"]
        return cs.Function("T_mapping", [T], [T_mapping])

    @staticmethod
    def setup_constraints(sym: cs.MX, low: NDArray, high: NDArray) -> cs.MX:
        dim = low.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-low, high))
        return A @ sym - b

    def select_action(self, obs: NDArray) -> NDArray:
        """Select the next action based on the current observation.

        Args:
            obs (NDArray): Current observation.

        Returns:
            NDArray: Selected action.
        """
        assert not self._requires_recompile, "Acados model must be recompiled"
        assert self.gaussian_process is not None, "Gaussian processes are not initialized"
        # Set initial condition (0-th state)
        self.acados_solver.set(0, "lbx", obs)
        self.acados_solver.set(0, "ubx", obs)
        nu = self.model.nu
        # Set the probabilistic state and input constraint limits
        state_constraint, input_constraint = self.propagate_constraint_limits()
        # Extend input constraints to T, set to zero at terminal step
        input_constraint = np.concatenate((input_constraint, np.zeros((2 * nu, 1))), axis=1)
        if self.sparse:
            p = np.concatenate((self.gp_state_input, state_constraint, input_constraint), axis=0)
        else:
            p = np.concatenate((state_constraint, input_constraint), axis=0)
        p = p.T.flatten()
        if (p_acados := self.acados_solver.get_flat("p")).shape != p.shape:
            raise ValueError(f"p shape mismatch: {p_acados.shape} != {p.shape}")
        self.acados_solver.set_flat("p", p)

        # Set reference for the control horizon
        goal_states = self.reference_trajectory()
        self.traj_step += 1
        y_ref = np.concatenate((goal_states[:, :-1], self.ref_action), axis=0)
        for idx in range(self.T):
            self.acados_solver.set(idx, "yref", y_ref[:, idx])
        self.acados_solver.set(self.T, "yref", goal_states[:, -1])

        # solve the optimization problem
        status = self.acados_solver.solve()
        assert status in [0, 2], f"acados returned unexpected status {status}."
        self.x_prev = self.acados_solver.get_flat("x").reshape(self.T + 1, -1).T
        self.u_prev = self.acados_solver.get_flat("u").reshape(self.T, -1).T
        return self.acados_solver.get(0, "u")

    def precompute_posterior_mean(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the posterior mean and for each GP."""
        ########################################################################
        # Task 2.3                                                             
        # TODO:                                                                
        # 1. For each trained GP in `self.gaussian_process`:                   
        #    a. Compute the posterior mean by solving the linear system        
        #       involving the GP's kernel matrix and training targets.         
        #    b. Concatenate the training input tensors for all GPs along the   
        #       last dimension.                                                
        # 3. Return both the stacked posterior means and the concatenated      
        #    training inputs as numpy arrays.                                  
        ########################################################################
        







        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return posterior_mean_numpy, train_data_numpy

    def precompute_sparse_posterior_mean(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the sparse posterior mean using the FITC approximation.

        Args:
            n_samples (int): Number of inducing points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sparse posterior means and sparse inputs as numpy arrays.
        """
        ########################################################################
        # Task 2.4                                                             
        # TODO:                                                                
        # 1. Use KMeans clustering to select `n_samples` inducing points from  
        #    the combined training inputs of all GPs. (use random_state = 0)   
        # 2. For each GP:                                                      
        #    a. Extract the relevant subset of the selected inducing points    
        #       corresponding to the GP's input dimensions.                    
        #    b. Compute the following matrices:                                
        #       - K_ss: Kernel matrix of inducing points.                      
        #       - K_xs: Kernel matrix between training inputs and inducing     
        #         points.                                                      
        #       - Gamma: Diagonal correction term for FITC.                    
        #       - Gamma_inv: Inverse of Gamma as a diagonal matrix.            
        #       - Sigma_inv: FITC covariance matrix for inducing points.       
        #    c. Compute the FITC sparse posterior mean for these inducing      
        #       points using the formula:                                      
        #       posterior_mean = torch.linalg.solve(Sigma_inv, K_xs.T) @       
        #                        Gamma_inv @ targets                           
        #    d. Store the posterior mean for this GP.
        # 3. Stack the sparse posterior means for all GPs and return them      
        #    together with the selected inducing points as numpy arrays.       
        ########################################################################

        inputs = torch.cat([gp.train_inputs[0] for gp in self.gaussian_process], dim=-1)
        targets = torch.stack([gp.train_targets for gp in self.gaussian_process], dim=1)
        kmeans_random_state = 0  # Random state for reproducibility

        




























        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return posterior_mean.numpy(force=True), sparse_inputs.numpy(force=True)

    def sparse_gp_kernels_cs(self, n_samples: int) -> List[cs.Function]:
        """Create CasADi functions for sparse GP kernels.

        Args:
            n_samples (int): Number of inducing points.

        Returns:
            List[ccs.Function]: CasADi kernel functions for each GP.
        """
        gps = self.gaussian_process

        lengthscales = [gp.covar_module.base_kernel.lengthscale.numpy(force=True) for gp in gps]
        scales = [gp.covar_module.outputscale.numpy(force=True) for gp in gps]
        Nx = sum([gp.train_inputs[0].shape[1] for gp in self.gaussian_process])

        # Create CasADI function for computing the kernel K_zs with parameters for z, s, length scales and output scaling. We need the CasADI version of this for symbolic differentiation in the MPC optimization.
        z1s = [cs.SX.sym("z1", gp.train_inputs[0].shape[1]) for gp in gps]
        z2 = cs.SX.sym("z2", n_samples, Nx)
        # Compute the kernel functions for each GP
        ks = [
            covSE_vectorized(z1s[i], z2[:, self.gp_idx[i]], lengthscales[i], scales[i])
            for i in range(len(gps))
        ]
        ks_fn = [
            cs.Function("K_s", [z1s[i], z2], [ks[i]], ["z1", "z2"], ["K"]) for i in range(len(gps))
        ]
        return ks_fn

    @torch.no_grad()
    def propagate_constraint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the state and input constraints forward using the GP dynamics.

        This function computes the state and input constraints for the MPC horizon based on the
        previous state and input trajectories, the GP dynamics, and the LQR gain.
        It uses the Gaussian processes to approximate the uncertainty in the dynamics and tightens
        the constraints accordingly.

        Returns:
            Tuple[np.ndarray, np.ndarray]: State and input constraints for the MPC horizon.
        """
        # 2 constraints per state and input. One to bound from below, one to bound from above.
        state_constraint = np.zeros((self.model.nx * 2, self.T + 1))
        input_constraint = np.zeros((self.model.nu * 2, self.T))

        if self.x_prev is None or self.u_prev is None:  # No previous rollout, return all zeros
            return state_constraint, input_constraint

        # cov_x can also be initialized with a diagonal matrix, i.e. np.eye(self.model.nx) * sigma
        cov_x = np.zeros((self.model.nx, self.model.nx))
        z = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))
        z_tensor = torch.tensor(z, device=self.device)

        covs_diag = []
        for gp, idx in zip(self.gaussian_process, self.gp_idx):
            gp.eval()
            with fast_pred_var(state=True), fast_pred_samples(state=True):
                cov = gp.likelihood(gp(z_tensor[:, idx])).covariance_matrix
            covs_diag.append(torch.diag(cov).numpy(force=True))

        idx_phi, idx_theta = 6, 7
        cos_phi_sin_theta_2 = np.cos(z[:, idx_phi]) * np.sin(z[:, idx_theta]) ** 2
        sin_phi_2 = (-np.sin(z[:, idx_phi])) ** 2
        cos_phi_cos_theta_2 = (np.cos(z[:, idx_phi]) * np.cos(z[:, idx_theta])) ** 2

        cov_d_batch = np.zeros((z.shape[0], 5, 5))
        cov_d_batch[:, 0, 0] = covs_diag[0] * cos_phi_sin_theta_2
        cov_d_batch[:, 1, 1] = covs_diag[0] * sin_phi_2
        cov_d_batch[:, 2, 2] = covs_diag[0] * cos_phi_cos_theta_2
        cov_d_batch[:, 3, 3] = covs_diag[1]
        cov_d_batch[:, 4, 4] = covs_diag[2]

        cov_noise = [gp.likelihood.noise.numpy(force=True) for gp in self.gaussian_process]
        cov_noise_batch = np.zeros((z.shape[0], 5, 5))
        cov_noise_batch[:, 0, 0] = cov_noise[0] * cos_phi_sin_theta_2
        cov_noise_batch[:, 1, 1] = cov_noise[0] * sin_phi_2
        cov_noise_batch[:, 2, 2] = cov_noise[0] * cos_phi_cos_theta_2
        cov_noise_batch[:, 3, 3] = cov_noise[1]
        cov_noise_batch[:, 4, 4] = cov_noise[2]

        # discretize
        cov_noise_batch = cov_noise_batch * self.dt**2
        cov_d_batch = cov_d_batch * self.dt**2

        # Assuming that the state constraints are of the form A * x <= b and that the first half of
        # the constraints are negative -> A = [-I, I]
        state_cnstr_A = np.concatenate([-np.eye(self.model.nx), np.eye(self.model.nx)], axis=0)
        input_cnstr_A = np.concatenate([-np.eye(self.model.nu), np.eye(self.model.nu)], axis=0)
        inv_cdf_A_abs_state = self.inverse_cdf * np.abs(state_cnstr_A)
        inv_cdf_A_abs_input = self.inverse_cdf * np.abs(input_cnstr_A)

        ########################################################################
        # Task 2.5                                                             
        # TODO:                                                                
        # For each time step in the MPC horizon:                               
        # 1. Compute the current state covariance and the corresponding input  
        #    covariance using the LQR gain (self.lqr_gain).                    
        # 2. For each GP, approximate the output covariance (including GP mean 
        #    and noise) at the current step.                                   
        # 3. Use the computed covariances to tighten (shrink) the state and    
        #    input constraints for probabilistic safety.                       
        # 4. Propagate the state covariance forward to the next time step      
        #    using the discrete-time dynamics and previous covariances.        
        #    The dynamics are given by self.discrete_dfdx, self.discrete_dfdu, 
        #    and self.Bd.                                                      
        ########################################################################
        


















        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        # Update Final covariance.
        state_constraint[:, -1] = -inv_cdf_A_abs_state @ np.sqrt(np.diag(cov_x))
        return state_constraint, input_constraint

    @staticmethod
    def setup_prior_dynamics(
        dfdx: NDArray, dfdu: NDArray, Q: NDArray, R: NDArray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the LQR gain used for propograting GP uncertainty from the prior dynamics."""
        A, B = discretize_linear_system(dfdx, dfdu, dt, exact=True)
        ########################################################################
        # Task 2.1                                                             
        # TODO:                                                                
        # 1. Solve the discrete algebraic Riccati equation (DARE) to find the  
        #   matrix optimal cost-to-go matrix. (Hint: see scipy.linalg package) 
        # 2. Compute the LQR gain
        ########################################################################
        





        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return A, B, lqr_gain

    def reference_trajectory(self) -> NDArray:
        """Get the reference trajectory (assumed to be periodic).

        Returns:
            NDArray: Reference trajectory.
        """
        indices = np.arange(self.traj_step, self.traj_step + self.T + 1) % self.traj.shape[-1]
        return self.traj[:, indices]  # (nx, T+1)

    def infer_gps_only(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer the GP mean and std for a batch of inputs.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input data, shape (N, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Means and stds, each shape (N, O).
        """
        assert self.gaussian_process is not None, "Gaussian processes are not initialized"
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
        means = []
        stds = []
        for gp, idx in zip(self.gaussian_process, self.gp_idx):
            mean, std = infer_gp(gp, x[:, idx], self.device)
            means.append(mean.unsqueeze(1) if mean.ndim == 1 else mean)
            stds.append(std.unsqueeze(1) if std.ndim == 1 else std)
        means = torch.cat(means, dim=1)
        stds = torch.cat(stds, dim=1)
        return means, stds

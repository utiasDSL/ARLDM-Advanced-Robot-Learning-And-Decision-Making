import dataclasses
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import casadi as cs
import gpytorch
import numpy as np
import torch
from exercise05.utils import select_inducing_points
from linear_operator.utils.cholesky import psd_safe_cholesky

# Constants
JITTER_BASE = 1e-6  # base jitter scale
MIN_LS = 1e-6  # minimum lengthscale for CasADi evals
MIN_DIAG_MEAN = 1e-8  # minimum mean(diag(K)) to scale jitter


def _scaled_jitter(K: torch.Tensor, base: float = JITTER_BASE) -> float:
    """Scale jitter by the average diagonal magnitude for numeric stability."""
    diag_mean = torch.clamp(K.diag().mean(), min=MIN_DIAG_MEAN)
    return float(base) * float(diag_mean)


def _safe_cholesky(K: torch.Tensor, base: float = JITTER_BASE) -> torch.Tensor:
    """Robust Cholesky using linear_operator.psd_safe_cholesky with scaled jitter."""
    return psd_safe_cholesky(K, jitter=_scaled_jitter(K, base))


# ---------- Low-level single output sparse variational GP ----------
class GaussianProcess(gpytorch.models.ExactGP):
    """Simple exact Gaussian Process model with zero mean and RBF kernel."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ard_num_dims: int = None,
        n_inducing_points: int = 20,
        **kwargs,  # noqa: ANN003
    ):
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        super().__init__(x, y, self.likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if ard_num_dims is None:
            ard_num_dims = x.shape[1]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-5),
            )
        )
        # Save dimensions for later use
        self.n_train, self.Din = x.shape
        self._K_tilde, self._L_tilde = None, None  # Only computed once the GP is trained
        n_inducing_points = min(n_inducing_points, self.n_train)
        if n_inducing_points == self.n_train:
            self.inducing_points = x.clone()
        else:
            self.inducing_points = select_inducing_points(x, n_inducing_points, mode="fps")
        self.n_inducing_points = self.inducing_points.shape[0]

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def _ensure_cholesky(self):
        """Ensure that the Cholesky factorization of K + sigma^2 I is computed and cached."""
        if self._L_tilde is not None:
            return
        X = self.train_inputs[0]
        with torch.no_grad():
            K_nn = self.covar_module(X).to_dense()
            noise = self.likelihood.noise
            n = K_nn.shape[0]
            K_tilde = K_nn + noise * torch.eye(n, device=K_nn.device, dtype=K_nn.dtype)
            L = _safe_cholesky(K_tilde)
        self._K_tilde = K_tilde
        self._L_tilde = L

    def compute_covariances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K(X,X)+sigma^2 I and a numerically stable inverse via Cholesky."""
        K = self.covar_module(self.train_inputs[0]).add_diagonal(self.likelihood.noise).to_dense()
        L = _safe_cholesky(K)
        K_inv = torch.cholesky_inverse(L)
        return K, K_inv

    def fit(self, X_train, Y_train, epochs: int = 100, batch_size: int = 256, lr: float = 1e-3):
        assert (
            X_train.ndim == 2
            and X_train.shape[1] == self.Din
            and (Y_train.ndim == 1 or Y_train.shape[1] == 1)
            and X_train.shape[0] == Y_train.shape[0]
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        _fit_gp(self, opt, mll, X_train, Y_train, epochs=epochs, batch_size=batch_size)
        # Invalidate cached factors (hyperparams changed)
        self._K_tilde, self._L_tilde = None, None

    def get_sparse_parameterization(self, z: cs.MX, id: int) -> Tuple[cs.MX, cs.MX]:
        """Get a sparse representation of the GP dynamics and its parameters.

        The parameter vector packs:
            [ alpha (n_inducing),
              inducing_inputs (n_inducing * Din) ]

        Args:
            z: CasADi MX variable of shape (Din,) representing the query input.
            id: Unique integer id for symbol naming across multiple GPs.

        Returns:
            sym_expr: CasADi MX scalar expression for GP predictive mean at z.
            sym_params: CasADi MX parameter vector containing (means, inducing points).
        """
        assert isinstance(z, cs.MX), "z must be a CasADi MX"
        assert z.shape[0] == self.Din, f"Expected z shape ({self.Din},), got {z.shape}"
        n, Din = self.n_inducing_points, self.Din
        sym_params = cs.MX.sym(f"gp{id}_params", n + n * Din)
        alpha = sym_params[:n]
        # Required variables
        # X: Symbolical inducing points (n, Din)
        # ls: Lengthscales (Din,)
        # sf2: Output scale (scalar)
        X = cs.reshape(sym_params[n:], Din, n).T  # (n, Din)
        ls = cs.DM(self.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1))
        sf2 = float(self.covar_module.outputscale.detach().cpu().numpy())
        ls = cs.fmax(ls, MIN_LS)  # avoid too small lengthscales for numerical stability
        #####################################################################
        # Task 1.2: Create a CasADi expression for the sparse GP predictive mean at z using the RBF kernel. You need: z, X, ls, sf2, alpha, and Din (for reshaping).
        #####################################################################
        mean_expr: cs.MX = None  # Compute this
        





        #######################################################################
        #                        END OF YOUR CODE
        #######################################################################
        assert mean_expr.shape == (1,), f"pred has wrong shape {mean_expr.shape}"
        assert sym_params.shape == (n + n * Din,), f"sym_params has wrong shape {sym_params.shape}"
        return mean_expr, sym_params

    def get_sparse_parameters(self) -> np.ndarray:
        """Get the numerical parameters of the trained GP for use in the sparse dynamics.

        Returns:
            params: [alpha (n_inducing), inducing_inputs_flat (n_inducing*Din)] where alpha = (K + sigma**2 I)^{-1} y
        """
        self._ensure_cholesky()

        # Required variables
        X_full = self.train_inputs[0]
        y = self.train_targets  # (n,)
        Z = self.inducing_points  # (m, Din)
        L_tilde = self._L_tilde  # cached Cholesky factor
        covar_module = self.covar_module  # kernel module

        with torch.no_grad():
            #####################################################################
            # Task 1.2: Compute the sparse GP parameters alpha
            # Hint: Use the _safe_cholesky function to compute the Cholesky factorization of K + σ²I
            # You need: L_tilde, y, Z, and covar_module
            #####################################################################
            alpha = None  # Compute this

            












            alpha = alpha.squeeze(-1).cpu().numpy()  # (m,)
            #######################################################################
            #                        END OF YOUR CODE
            #######################################################################
        Z_flat = Z.detach().cpu().numpy().reshape(-1)
        assert alpha.shape == (self.n_inducing_points,), (
            f"Expected alpha shape {(self.n_inducing_points,)}, got {alpha.shape}"
        )
        assert isinstance(alpha, np.ndarray), "alpha must be a numpy array"
        assert Z_flat.shape == (self.n_inducing_points * self.Din,), (
            f"Expected Z_flat shape {(self.n_inducing_points * self.Din,)}, got {Z_flat.shape}"
        )
        return np.concatenate([alpha, Z_flat], axis=0)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and variance at x.

        Returns:
            mean: Predictive mean at x.
            variance: Predictive variance at x.
        """
        x = torch.tensor(x, device=next(self.parameters()).device, dtype=torch.float32)
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x))
            return pred.mean.cpu(), pred.variance.cpu()


class _SingleOutputSVGP(gpytorch.models.ApproximateGP):
    """Single-output Sparse Variational GP model."""

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, learn_inducing: bool = True, verbosity: str = "info"
    ):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        self.logger = logging.getLogger(self.__class__.__name__)
        log_level = getattr(logging, verbosity.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        self.n_inducing_points, self.Din = x.shape

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.n_inducing_points
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            x,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
            jitter_val=1e-4,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=self.Din, lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-5)
            )
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=next(self.parameters()).device, dtype=torch.float32)
        else:
            x = x.detach().to(device=next(self.parameters()).device, dtype=torch.float32)
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x))
            return pred.mean.cpu(), pred.variance.cpu()

    def get_sparse_parameterization(self, z: cs.MX, id: int = 42) -> Tuple[cs.MX, cs.MX]:
        """Get a sparse representation of the GP dynamics and its parameters.

        Same convention as the exact GP:
            params = [alpha (n), inducing_inputs (n * Din)]

        Args:
            z: Symbolical inputs of shape (Din,).
            id: Integer id to keep symbol names unique.

        Returns:
            sym_expr: Scalar MX predictive mean at z.
            sym_params: MX parameter vector.
        """
        assert isinstance(z, cs.MX), "z must be a CasADi MX"
        assert z.shape[0] == self.Din, f"Expected z shape ({self.Din},), got {z.shape}"

        n, Din = self.n_inducing_points, self.Din
        sym_params = cs.MX.sym(f"svgp{id}_params", n + n * Din)
        alpha = sym_params[:n]
        # Required variables
        # X: Inducing points (n, Din)
        # ls: Lengthscales (Din,)
        # sf2: Output scale (scalar)
        X = cs.reshape(sym_params[n:], Din, n).T
        ls = cs.DM(self.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1))
        sf2 = float(self.covar_module.outputscale.detach().cpu().numpy())
        ls_safe = cs.fmax(ls, MIN_LS)  # Guard very small lengthscales
        #####################################################################
        # Task 1.2: Create a CasADi expression for the sparse GP predictive mean at z using the RBF kernel. You need: z, X, ls_safe, sf2, alpha, and Din.
        #####################################################################
        mean_expr: cs.MX = None  # Compute this
        





        ######################################################################
        #                           END OF YOUR CODE
        ######################################################################
        assert mean_expr.shape[0] == 1, f"Expected only one output, got {mean_expr.numel()}"
        assert sym_params.shape[0] == n + n * Din, (
            f"Expected sym_params shape {(n + n * Din,)}, got {sym_params.shape}"
        )
        return mean_expr, sym_params

    def get_sparse_parameters(self) -> np.ndarray:
        """Get the numerical parameters of the trained GP for use in the sparse dynamics.

        Returns:
            params: [alpha (n_inducing), inducing_inputs_flat]
                    alpha = K_ZZ^{-1} m_u (variational posterior)
        """
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            #####################################################################
            # Task 1.2: Compute alpha, the sparse GP parameters.
            # Hint: Use the _safe_cholesky function to compute the Cholesky factorization of K_ZZ
            # You need: K_ZZ and mean (variational posterior).
            #####################################################################
            Z = self.variational_strategy.inducing_points # (n_inducing, Din)
            K_ZZ = self.covar_module(Z).to_dense() # (n_inducing, n_inducing)
            mean = self(self.variational_strategy.inducing_points).mean # (n_inducing,)
            alpha = None  # Compute this
            




            alpha = alpha.squeeze(-1).cpu().numpy()
            ######################################################################
            #                           END OF YOUR CODE
            ######################################################################
            assert alpha.shape[0] == self.n_inducing_points, "alpha has wrong shape"
            X_flat = Z.cpu().numpy().reshape(-1)
        return np.concatenate([alpha, X_flat], axis=0)

    def compute_covariances(self):
        K = self.covar_module(self.train_inputs[0]).add_diagonal(self.likelihood.noise).to_dense()
        L = _safe_cholesky(K)
        K_inv = torch.cholesky_inverse(L)
        return K, K_inv

    def fit(
        self,
        X_train,
        Y_train,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
        rtol: float = 1e-4,
    ):
        assert (
            X_train.ndim == 2
            and X_train.shape[1] == self.Din
            and (Y_train.ndim == 1 or Y_train.shape[1] == 1)
            and X_train.shape[0] == Y_train.shape[0]
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=Y_train.numel())
        _fit_gp(
            self,
            opt,
            elbo,
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            rtol=rtol,
            logger=self.logger,
        )


@dataclass
class ResidualGPConfig:
    gp_type: str = "svgp"
    n_inducing_points: int = 20
    learn_inducing: bool = True
    seed: Optional[int] = None  # reproducibility for inducing selection
    device: str = "cpu"


@dataclass
class ResidualGPTrainingConfig:
    lr: float = 1e-4  # lowered for stability
    epochs: int = 100
    batch_size: int = 256
    patience: int = 20
    rtol: float = 1e-4


class ResidualGP:
    """Residual Gaussian Process model. Uses multiple single-output GPs which input dimensions are defined by group_inputs.

    The outputs are mapped to the full state residual vector by assuming independence between the GPs.
    The default configuration is for a quadrotor attitude model with 3 GPs:
    - GP1: thrust command -> thrust residual
    - GP2: (roll angle, roll rate, roll command) -> roll rate residual
    - GP3: (pitch angle, pitch rate, pitch command) -> pitch rate residuseedal
    The thrust residual is then mapped to accelerations in x,y,z by assuming small angles.

    Args:
        X: (N, nx+nu) training inputs
        Y: (N, n_gp) training targets (n_gp = number of GPs)
        group_inputs: List of input index lists for each GP
        gp_cfg: dict of ResidualGPConfig parameters
        train_cfg: dict of ResidualGPTrainingConfig parameters
    """

    def __init__(
        self,
        X_list: list[torch.Tensor],
        Y_list: list[torch.Tensor],
        gp_cfg: Union[dict, ResidualGPConfig] = {},
        train_cfg: Union[dict, ResidualGPTrainingConfig] = {},
        verbosity: str = "info",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        log_level = getattr(logging, verbosity.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        self.train_cfg = (
            ResidualGPTrainingConfig(**train_cfg) if isinstance(train_cfg, dict) else train_cfg
        )
        self.cfg = ResidualGPConfig(**gp_cfg) if isinstance(gp_cfg, dict) else gp_cfg
        self.device = "cpu" if not torch.cuda.is_available() else self.cfg.device
        self.n_gps = len(X_list)

        self._parameter_cache = None
        self.setup_gps(X_list, Y_list, verbosity)

    def setup_gps(self, X_list, Y_list, verbosity: str = "info"):
        self._gps: List[_SingleOutputSVGP] = []
        for x, y in zip(X_list, Y_list):
            if self.cfg.gp_type.lower() == "svgp":
                x = select_inducing_points(
                    x, self.cfg.n_inducing_points, mode="fps", seed=self.cfg.seed
                )
                gp = _SingleOutputSVGP(
                    x, y, learn_inducing=self.cfg.learn_inducing, verbosity=verbosity
                ).to(self.device)
            else:
                gp = GaussianProcess(
                    x,
                    y,
                    ard_num_dims=None,
                    verbosity=verbosity,
                    n_inducing_points=self.cfg.n_inducing_points,
                ).to(self.device)
            self._gps.append(gp)

    # --------------- Training ---------------

    def fit(self, X_list, Y_list):
        assert len(X_list) == len(Y_list) == self.n_gps == len(self._gps), (
            "Input/output data must match number of GPs."
        )
        for gp, Xg, yg in zip(self._gps, X_list, Y_list):
            assert isinstance(Xg, torch.Tensor) and isinstance(yg, torch.Tensor), (
                "X and Y must be torch.Tensors"
            )
            assert Xg.shape[1] == gp.Din, f"X has wrong input dimension {Xg.shape[1]} != {gp.Din}"
            assert yg.ndim == 1 or yg.shape[1] == 1, "Y must be 1D or 2D with one column"
            gp.fit(Xg, yg, **dataclasses.asdict(self.train_cfg))

        self._parameter_cache = self._get_sparse_parameters()

    # ------------- Export / parameter update -------------

    def _get_sparse_parameters(self):
        return [gp.get_sparse_parameters() for gp in self._gps]

    def update_residual_parameters(self) -> np.ndarray:
        if self._parameter_cache is None:
            self._parameter_cache = self._get_sparse_parameters()
        return np.concatenate(self._parameter_cache, axis=0)

    # ------------- CasADi symbolic residual dynamics -------------

    def get_sparse_parameterization(self, Z: list[cs.MX]) -> Tuple[list[cs.MX], list[cs.MX]]:
        """Get CasADi symbolic expression for the residual dynamics and required GP parameters.

        Args:
            Z: List of CasADi MX variables where each entry corresponds to the input of one GP.
                The ordering must match the one used in setup_gps.

        Returns:
            residual_expr: List of CasADi MX expressions for the individual GP outputs.
            gp_params: List of CasADi MX variables for the GP parameters (in order of export_params).
        """
        assert isinstance(Z, Sequence) and len(Z) == self.n_gps, (
            "Z must be a list of CasADi MX variables"
        )
        residual_expr = []
        gp_params = []

        for i, (z, gp) in enumerate(zip(Z, self._gps)):
            assert isinstance(z, cs.MX), "Z must be a list of CasADi MX variables"
            gp_sym_expr, gp_sym_params = gp.get_sparse_parameterization(z, i)
            residual_expr.append(gp_sym_expr)
            gp_params.append(gp_sym_params)
        return residual_expr, gp_params


def _fit_gp(
    gp: GaussianProcess,
    opt: torch.optim.Optimizer,
    loss_fcn,
    X_train,
    y_train,
    epochs=1000,
    batch_size=256,
    patience=20,
    rtol=1e-4,
    X_val=None,
    y_val=None,
    logger: logging.Logger = None,
    log_epochs: int = 10,
):
    """Unified training loop to train a Gaussian Process model with optional early stopping using a relative improvement criterion.

    Args:
        gp: The Gaussian Process model to be trained.
        opt: The optimizer to use for training.
        loss_fcn: The loss function to minimize.
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        rtol (float): Relative tolerance for early stopping.
        X_val (torch.Tensor, optional): Validation input data for early stopping.
        y_val (torch.Tensor, optional): Validation target data for early stopping.
        logger (logging.Logger, optional): Logger for debug information.
        log_epochs (int): Number of epochs between logging training status.
    """
    device = next(gp.parameters()).device
    gp.train()
    gp.likelihood.train()
    ds = torch.utils.data.TensorDataset(X_train, y_train.squeeze(-1))
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    patience_counter = 0
    best_val_loss = np.inf
    if X_val is not None and y_val is not None:
        ds_val = torch.utils.data.TensorDataset(X_val, y_val)
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=min(batch_size, len(ds_val)), shuffle=False
        )
    else:
        dl_val = None

    if logger is not None:
        logger.setLevel(logging.DEBUG)

    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            out = gp(xb.to(device))
            # Gpytorch loss function requires negative sign
            loss = -loss_fcn(out, yb.to(device))
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(ds)
        if dl_val is not None:
            gp.eval()
            gp.likelihood.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xb, yb in dl_val:
                    out = gp(xb.to(device))
                    val_loss += -loss_fcn(out, yb.to(device)).item()
                val_loss /= len(ds_val)
            gp.train()
            gp.likelihood.train()
        val_loss = val_loss if dl_val is not None else ep_loss
        if val_loss >= best_val_loss - rtol * abs(best_val_loss):
            patience_counter += 1
            if patience_counter >= patience:
                if logger is not None:
                    logger.info(
                        f"Early stopping after {ep} epochs. Best val loss: {best_val_loss:.4f}"
                    )
                break
        else:
            patience_counter = 0
            best_val_loss = val_loss if dl_val is not None else ep_loss
        if ep % max(1, epochs // log_epochs) == 0 and logger is not None:
            if dl_val is not None:
                logger.debug(f"[GP Train] ep={ep} train_loss={ep_loss:.4f} val_loss={val_loss:.4f}")
            else:
                logger.debug(f"[GP Train] ep={ep} train_loss={ep_loss:.4f}")
    gp.eval()
    gp.likelihood.eval()

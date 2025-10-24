"""Utility functions for Gaussian Processes."""

from typing import Union

import casadi as cs
import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean


def covSE_single(x, z, ell, sf2):
    diff = (x - z) / ell.T  # Normalize the difference by the lengthscale
    dist = cs.sum1(diff**2)  # Compute the squared Euclidean distance
    return sf2 * cs.exp(-0.5 * dist)


def covSE_vectorized(x, Z, ell, sf2):
    """Vectorized kernel version of covSE_single."""
    x_reshaped = cs.repmat(x, 1, Z.shape[0])  # Reshape x to match Z's dimensions for broadcasting
    diff = (x_reshaped - Z.T) / ell.T
    dist = cs.sum1(diff**2)
    return sf2 * cs.exp(-0.5 * dist)


class GaussianProcess(gpytorch.models.ExactGP):
    """Gaussian Process decorator for gpytorch."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, ard_num_dims: int = None):
        """Initialize Gaussian Process."""
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
        likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        super().__init__(x, y, likelihood)
        self.mean_module = ConstantMean()
        # if ard_num_dims is None:
        #     ard_num_dims = x.shape[1] if x.ndim > 1 else 1
        # self.covar_module = ScaleKernel(RBFKernel())
        self.covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-5),
            )
        )
        # Save dimensions for later use
        self.n_ind_points = self.train_inputs[0].shape[0]
        self.input_dimension = self.train_inputs[0].shape[1]
        self.K, self.K_inv = None, None  # Only computed once the GP is trained

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def compute_covariances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K(X,X) + sigma*I and its inverse."""
        K = self.covar_module(self.train_inputs[0]).add_diagonal(self.likelihood.noise).to_dense()
        return K, torch.linalg.inv(K)


def fit_gp(
    gp: GaussianProcess,
    n_train: int = 500,
    lr: float = 0.01,
    device: str = "cpu",
    patience: int = 5,
    rtol: float = 1e-3,
    val_x: Union[torch.Tensor, None] = None,
    val_y: Union[torch.Tensor, None] = None,
):
    """Fit a GP to its training data with basic early stopping.

    Args:
        gp (GaussianProcess): The Gaussian Process model to fit.
        n_train (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        device (str): Device to use for training ('cpu' or 'cuda').
        patience (int): Number of epochs with no improvement after which training will be stopped.
        rtol (float): Relative tolerance for early stopping.
        val_x (torch.Tensor, optional): Validation inputs for early stopping.
        val_y (torch.Tensor, optional): Validation targets for early stopping.
    """
    assert isinstance(gp, GaussianProcess), f"gp must be a GaussianProcess, got {type(gp)}"
    train_x = gp.train_inputs[0].to(device)
    train_y = gp.train_targets.to(device)
    if val_x is not None and val_y is not None:
        val_x = val_x.to(device)
        val_y = val_y.to(device)

    gp.train().to(device)

    ########################################################################
    # Task 1.2
    # TODO:
    # 1. Set up the optimizer and the marginal log likelihood loss.
    # 2. Implement a basic training loop over 'n_train' epochs with simple
    # early stopping (if no validation data is given use the relative loss
    # improvement per iteration (at least 'r_tol').
    # 3. Use the ExactMarginalLogLikelihood loss
    ########################################################################
    









































    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################
    gp.K, gp.K_inv = gp.compute_covariances()


def infer_gp(gp: GaussianProcess, x: torch.Tensor, device="cpu") -> torch.Tensor:
    """Infer the mean of the GP at a given point."""
    assert isinstance(gp, GaussianProcess), f"gp must be a GaussianProcess, got {type(gp)}"
    assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
    gp.eval().to(device)
    x = x.to(device)
    with torch.no_grad():
        pred = gp.likelihood(gp(x))
        mean = pred.mean
        std = pred.stddev
    return mean.cpu(), std.cpu()


def compute_loss_gp(
    gp: GaussianProcess,
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    device="cpu",
) -> Union[torch.Tensor, np.ndarray]:
    assert isinstance(gp, GaussianProcess), f"gp must be a GaussianProcess, got {type(gp)}"
    assert isinstance(x, (torch.Tensor, np.ndarray)), "x must be a torch.Tensor or np.ndarray"
    assert isinstance(y, (torch.Tensor, np.ndarray)), "y must be a torch.Tensor or np.ndarray"
    assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
    assert gp.train_inputs[0].shape[1] == x.shape[1], (
        "x must have the same number of features as the GP's training data"
    )
    x = torch.FloatTensor(x).to(device) if isinstance(x, np.ndarray) else x.to(device)
    y = torch.FloatTensor(y).to(device) if isinstance(y, np.ndarray) else y.to(device)
    mean, _ = infer_gp(gp, x, device)
    loss = torch.nn.functional.mse_loss(mean, y, reduction="mean")
    return loss.item() if isinstance(loss, torch.Tensor) else loss


def gpytorch_predict2casadi(gp: GaussianProcess) -> cs.Function:
    """Convert the prediction function of a gpytorch model to casadi model."""
    assert isinstance(gp, GaussianProcess), f"Expected a GaussianProcess, got {type(gp)}"
    train_inputs = gp.train_inputs[0].numpy(force=True)
    train_targets = gp.train_targets.numpy(force=True)
    assert train_inputs.ndim == 2, "train_inputs must be a 2D array"
    lengthscale = gp.covar_module.base_kernel.lengthscale.to_dense().numpy(force=True)
    output_scale = gp.covar_module.outputscale.to_dense().numpy(force=True)

    z = cs.SX.sym("z", train_inputs.shape[1])
    kernel_fn = covSE_single(z, train_inputs.T, lengthscale.T, output_scale)
    K_xz = cs.Function("K_xz", [z], [kernel_fn], ["z"], ["K"])
    K_xx_inv = gp.K_inv.numpy(force=True)
    return cs.Function("pred", [z], [K_xz(z=z)["K"] @ K_xx_inv @ train_targets], ["z"], ["mean"])

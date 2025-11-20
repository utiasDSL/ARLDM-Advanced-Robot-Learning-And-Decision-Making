import time
from typing import Callable, Optional, Tuple, Union

import gpytorch as gpy
import numpy as np
import torch
from exercise04.utils import select_inducing_points, to_torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset


class BaseGaussianProcess:
    def __init__(self, kernel: str = "rbf", kernel_params: dict = None, noise: float = 1e-8, X_train=None, y_train=None, max_samples: Optional[int] = 1000):
        """Basic Gaussian Process implementation with different kernel options. Parameters for kernels must be provided as a dictionary.

        Args:
            kernel (str): Kernel type, options are "rbf", "linear", "poly", "matern", "exp".
            kernel_params (dict): Parameters for the chosen kernel.
            noise (float): Noise level for regularization.
            X_train (np.ndarray, optional): Training inputs. Defaults to None.
            y_train (np.ndarray, optional): Training targets. Defaults to None.
            max_samples (int, optional): Maximum number of samples to use for training. Defaults to 1000.
        """
        self.kernel_params = kernel_params or {"length_scale": 1.0}
        self.noise = noise
        self.max_samples = max_samples
        assert isinstance(X_train, (type(None), np.ndarray)) and isinstance(y_train, (type(None), np.ndarray)), "X_train and y_train must be numpy arrays or None."
        self.X_train = X_train
        if y_train and len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        self.y_train = y_train
        self.output_dim = None if y_train is None else y_train.shape[1]
        self.set_kernel(kernel)


    def set_kernel(self, kernel_type: str, kernel_params: dict = None):
        """Set the kernel and its parameters.

        Args:
            kernel_type (str): Kernel type, options are "rbf", "linear", "poly", "matern", "exp".
            kernel_params (dict): Parameters for the chosen kernel.
        """
        if not hasattr(self, f"{kernel_type}_kernel"):
            raise ValueError(f"Kernel '{kernel_type}' not supported.")
        self.kernel = getattr(self, f"{kernel_type}_kernel")
        assert isinstance(self.kernel, Callable)
        if kernel_params is not None:
            self.kernel_params = kernel_params

    def set_kernel_params(self, **kwargs : dict[str, any]):
        """Set kernel parameters.

        Args:
            **kwargs: Kernel parameters to set.
        """
        self.kernel_params.update(kwargs)

    def rbf_kernel(self, X1, X2):
        """Radial Basis Function (RBF) kernel.

        Inputs:
            X1, X2 (np.ndarray): Input arrays.

        Returns:
            np.ndarray: Kernel matrix.
        """
        ########################################################################
        # Task 1.1
        # TODO:
        # 1. Define distance measure
        # 2. Define rbf_kernel
        # Hints:
        # 1. Use cdist from scipy.spatial.distance to compute the distance
        # 2. Don't forget the length_scale
        # 3. IMPORTANT The rbf kernel is not equal to the exponential kernel.
        # You can find the equation for the rbf kernel for instance in
        # wikipedia.
        # https://en.wikipedia.org/wiki/Radial_basis_function_kernel.
        # The variance (σ) is also called length scale.
        ########################################################################
        length_scale = self.kernel_params.get("length_scale", 1.0)

        




        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return rbf_kernel

    def linear_kernel(self, X1, X2):
        """Linear kernel.

        Inputs:
            X1, X2 (np.ndarray): Input arrays.

        Returns:
            np.ndarray: Kernel matrix.
        """
        ########################################################################
        # Task 1.1
        # TODO:
        # 1. Implement linear kernel function
        ########################################################################
        if self.X_train is not None:
            c = np.mean(self.X_train, axis=0)
        else:
            c = self.kernel_params.get("c", 0)
        output_std = self.kernel_params.get("output_std", 1)
        std_b = self.kernel_params.get("std_b", 0)
        



        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return linear_kernel

    def polynomial_kernel(self, X1, X2):
        """Polynomial kernel.

        Inputs:
            X1, X2 (np.ndarray): Input arrays.

        Returns:
            np.ndarray: Kernel matrix.
        """
        ########################################################################
        # Task 1.1
        # TODO:
        # 1. Implement polynomial kernel function
        ########################################################################
        degree = self.kernel_params.get("degree", 2)
        coef0 = self.kernel_params.get("coef0", 1)
        



        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return poly_kernel

    def matern_kernel(self, X1, X2):
        """Matern kernel.

        Inputs:
            X1, X2 (np.ndarray): Input arrays.

        Returns:
            np.ndarray: Kernel matrix.
        """
        ########################################################################
        # Task 1.1
        # TODO:
        # 1. Define distance measure
        # 2. Implement Matern kernel function
        #    Use nu=0.5, 1.5, or 2.5 (see Wikipedia for formulas)
        # Hints:
        # 1. Use cdist from scipy.spatial.distance to compute the distance
        ########################################################################
        length_scale = self.kernel_params.get("length_scale", 1.0)
        nu = self.kernel_params.get("nu", 1.5)
        















        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return matern_kernel

    def exp_kernel(self, X1, X2):
        """Exponential kernel.

        Inputs:
            X1, X2 (np.ndarray): Input arrays.

        Returns:
            np.ndarray: Kernel matrix.
        """
        ########################################################################
        # Task 1.1
        # TODO:
        # 1. Define distance measure
        # 2. Implement exponential kernel function
        # Hints:
        # 1. Use cdist from scipy.spatial.distance to compute the distance
        ########################################################################
        length_scale = self.kernel_params.get("length_scale", 1.0)
        




        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return exp_kernel

    def fit(self, X, y):
        """Fit the Gaussian Process model.

        Inputs:
            X (np.ndarray): Training inputs.
            y (np.ndarray): Training targets.
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays."
        self.X_train = X
        self.y_train = y
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.output_dim = y.shape[1]

        if self.max_samples and X.shape[0] > self.max_samples:
            idx = np.random.choice(X.shape[0], self.max_samples, replace=False)
            X = X[idx]
            y = y[idx]
            self.X_train_subset = X # for prediction
        ########################################################################
        # Task 1.2
        # TODO:
        # 1. Compute K, the kernel matrix
        # 2. Compute L, the Cholesky decomposition, using np.linalg.cholesky and noise for regularization
        # 3. Compute alphas, the dual coefficients or weight vectors alphas for prediction
        # Hints:
        # 1. For multi-output, compute alphas for each output
        ########################################################################
        # Show relevant variables
        noise = self.noise  
        kernel = self.kernel
        output_dim = self.output_dim
        self.K, self.L, self.alphas = None, None, None  # Compute those
        ########################################################################

        







        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        self.K, self.L, self.alphas = K, L, alphas  # To show relevant variables

    def predict(self, X_test, return_std=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with uncertainty estimates. Assumes uncorrelated outputs.

        Inputs:
            X_test (np.ndarray): Test inputs.
            return_std (bool): If True, return standard deviation along with mean.

        Returns:
            mean (np.ndarray): Mean predictions.
            std (np.ndarray, optional): Standard deviations.
        """
        ########################################################################
        # Task 1.2
        # TODO:
        # 1. Calculate K_star, the kernel between X_test and X_train
        # 2. Use the dual coefficents alphas to compute the mean prediction
        # 3. If return_std is True, compute the standard deviation using K_star
        #    and L, and return both mean and std. Otherwise, mean will be
        #    returned by default.
        ########################################################################
        # Show relevant variables
        L = self.L
        X_train = self.X_train if not hasattr(self, "X_train_subset") else self.X_train_subset
        alphas = self.alphas
        kernel = self.kernel
        output_dim = self.output_dim
        mean, std = None, None  # Compute those
        ########################################################################
        












        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return mean


# Single-output GP is just a special case of BaseGaussianProcess
class GaussianProcess(BaseGaussianProcess):
    pass


# MultiOutputGaussianProcess is now identical to BaseGaussianProcess
class MultiOutputGaussianProcess(BaseGaussianProcess):
    pass


class IndependentMultitaskSVGPModel(gpy.models.ApproximateGP):
    def __init__(self, X_train, num_tasks, num_inducing_points):
        # Select inducing point for each task
        num_inducing_points = min(num_inducing_points, X_train.shape[0])
        inducing_points = torch.stack(
            [select_inducing_points(X_train, num_inducing_points, mode="kmeans", seed=i) for i in range(num_tasks)],
            dim=0,
        )  # (num_tasks, num_inducing_points, num_features)
        batch_shape = torch.Size([num_tasks])

        #######################################################################
        # Task 5.1
        # TODO:
        # 1. Define a CholeskyVariationalDistribution for each task (batch shape).
        # 2. Wrap it in an IndependentMultitaskVariationalStrategy.
        #######################################################################
        learn_inducing_locations = True  # To show relevant parameter
        variational_distribution, variational_strategy = None, None  # Define those  
        #######################################################################
        











        #######################################################################
        #                        END OF YOUR CODE
        #######################################################################

        self.var_distribution_type =variational_distribution.__class__  # For testing purposes
        self.var_strategy_type = variational_strategy.__class__  # For testing purposes

        super().__init__(variational_strategy)

        #######################################################################
        # Task 5.2
        # TODO:
        # 1. Define the mean and covariance modules
        # Hints:
        # 1. Use ConstantMean and ScaleKernel with RBFKernel
        #######################################################################
        batch_shape = batch_shape # To show relevant parameter
        mean_module = None
        covar_module = None
        #######################################################################
        





        #######################################################################
        #                        END OF YOUR CODE
        #######################################################################
        self.mean_module, self.covar_module = mean_module, covar_module

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPTrainer:
    def __init__(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        device,
        inducing_points_per_task: Optional[int] = 30,
        max_total_inducing_points: Optional[int] = 500,
    ):
        if device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cuda = True
        else:
            self.device = "cpu"
            self.cuda = False

        self.X_train = to_torch(X_train, device=self.device, dtype=torch.float32)
        self.y_train = to_torch(y_train, device=self.device, dtype=torch.float32)

        num_tasks = y_train.shape[1]
        num_inducing_points = min(
            y_train.shape[0], inducing_points_per_task, int(max_total_inducing_points // num_tasks)
        )

        self.model = IndependentMultitaskSVGPModel(self.X_train, num_tasks, num_inducing_points)
        self.likelihood = gpy.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        if self.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def train(self, epochs=30, lr=0.1, batch_size=128, log_interval=5):
        self.model.train()
        self.likelihood.train()
        params = [{"params": self.model.parameters()}, {"params": self.likelihood.parameters()}]
        optimizer = torch.optim.Adam(params, lr=lr)
        mll = gpy.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.y_train.size(0))
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print("Training SVGP model on device:", self.device)
        start_time = time.time()
        for i in range(epochs):
            loss_epoch = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            if (i + 1) % log_interval == 0 or i == 0 or i == epochs - 1:
                print(f"Iter {i + 1}/{epochs} - Loss: {loss_epoch / len(train_loader):.4f}")
        self.training_time = time.time() - start_time

    def infer(self, X_test: Union[np.ndarray, torch.Tensor], return_std=True, return_ci=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_test = to_torch(X_test, device=self.device, dtype=torch.float32)

        self.model.eval()
        self.likelihood.eval()
        start_time = time.time()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            pred: gpy.likelihoods.MultitaskGaussianLikelihood = self.likelihood(self.model(X_test))
        self.inference_time = time.time() - start_time
        mean = pred.mean.cpu().numpy()
        std = pred.variance.sqrt().cpu().numpy()
        lower, upper = pred.confidence_region()
        if return_std and return_ci:
            return mean, std, lower.cpu().numpy(), upper.cpu().numpy()
        elif return_std:
            return mean, std
        elif return_ci:
            return mean, lower.cpu().numpy(), upper.cpu().numpy()
        else:
            return mean,

import time

import gpytorch as gpy
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset


class GaussianProcess:
    def __init__(self, kernel: str = "rbf", length_scale: float = 1.0, noise: float = 1e-8):
        if kernel == "linear":
            self.kernel = self.linear_kernel
        elif kernel == "rbf":
            self.kernel = self.rbf_kernel
        else:
            raise ValueError("Invalid kernel. Choose 'linear' or 'rbf'.")

        self.length_scale = length_scale
        self.output_std = 1  # hyperparameter
        self.c = 0  # horizontal shift, hyperparameter
        self.std_b = 0  # vertical shift, hyperparameter
        self.noise = noise
        self.X_train = None
        self.y_train = None

    def linear_kernel(self, X1, X2):
        """Linear kernel."""
        return self.std_b + self.output_std * (X1 - self.c) @ (X2 - self.c).T

    def rbf_kernel(self, X1, X2):
        """Radial Basis Function (RBF) kernel."""
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
        






        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return rbf_kernel

    def fit(self, X, y):
        """Fit the Gaussian Process model."""
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(X, X)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(len(X)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

    def predict(self, X_test, return_std=False):
        """Make predictions with uncertainty estimates."""
        ########################################################################
        # Task 1.2                                                             
        # TODO:                                                                
        # 1. Calculate K_star, the kernel between X_test and self.X_train      
        # 2. Use self.alpha to compute the mean prediction                     
        # 3. If return_std is True, compute the standard deviation using K_star
        # and self.L, and return both mean and std. Otherwise, mean will be    
        # returned by default.                                                 
        ########################################################################
        










        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return mean


class MultiOutputGaussianProcess:
    def __init__(self, kernel: str = "rbf", length_scale: float = 1.0, noise: float = 1e-8):
        self.kernel = kernel  # Not used in this implementation, but kept for interface consistency
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.output_dim = None
        self.models = []  # Will store alpha values for each output dimension

    def rbf_kernel(self, X1, X2):
        """Radial Basis Function (RBF) kernel."""
        ########################################################################
        # Task 1.1                                                              
        # TODO:                                                                
        # 1. Define distance measure                                           
        # 2. Define rbf_kernel                                                 
        # Hints:                                                               
        # 1. Use cdist from scipy.spatial.distance to compute the distance     
        ########################################################################
        






        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return rbf_kernel

    def fit(self, X, y):
        """Fit the Gaussian Process model with multi-dimensional outputs."""
        self.X_train = X
        self.y_train = y

        # Handle both 1D and multi-dimensional outputs
        if len(y.shape) == 1:
            self.y_train = y.reshape(-1, 1)

        self.output_dim = self.y_train.shape[1]

        # Compute kernel matrix once (shared across all output dimensions)
        self.K = self.rbf_kernel(X, X)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(len(X)))

        # Compute alpha for each output dimension
        self.alphas = np.zeros((len(X), self.output_dim))
        for i in range(self.output_dim):
            self.alphas[:, i] = np.linalg.solve(
                self.L.T, np.linalg.solve(self.L, self.y_train[:, i])
            )

    def predict(self, X_test, return_std=False):
        """Make predictions with uncertainty estimates for each output dimension."""
        ########################################################################
        # Task 1.2                                                             
        # TODO:                                                                
        # 1. Calculate K_star, the kernel between X_test and X_train           
        # 2. Use self.alphas to compute the mean prediction                    
        # 3. If return_std is True, compute the standard deviation using K_star
        # and self.L, and return both mean and std. Otherwise, mean will be    
        # returned by default.                                                 
        ########################################################################
        



















        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        return mean


class IndependentMultitaskSVGPModel(gpy.models.ApproximateGP):
    def __init__(self, X_train, num_tasks, num_inducing_points):
        # We select different inducing points for each task by randomly selecting them from the training data
        # print("Shape of X_train:", X_train.shape)
        num_features = X_train[0].shape[-1]
        inducing_points = torch.zeros(num_tasks, num_inducing_points, num_features)
        print("Inducing points shape:", inducing_points.shape)
        for i in range(num_tasks):
            random_indices = torch.randperm(X_train.shape[0])[:num_inducing_points]
            # print("Random indices:", random_indices)
            inducing_points[i, :, :] = X_train[random_indices, :]
            # print("Inducing points for task", i, ":", inducing_points[i, :, :])
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpy.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpy.variational.IndependentMultitaskVariationalStrategy(
            gpy.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpy.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPTrainer:
    def __init__(self, X_train, y_train, device):
        if device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cuda = True
        else:
            self.device = "cpu"
            self.cuda = False

        if isinstance(X_train, np.ndarray):
            self.X_train = torch.FloatTensor(X_train).to(self.device)
        else:
            self.X_train = X_train.to(self.device)
        if isinstance(y_train, np.ndarray):
            self.y_train = torch.FloatTensor(y_train).to(self.device)
        else:
            self.y_train = y_train.to(self.device)

        num_tasks = y_train.shape[1]
        num_inducing_points = int(500 // num_tasks)
        self.model = IndependentMultitaskSVGPModel(self.X_train, num_tasks, num_inducing_points)
        self.likelihood = gpy.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        if self.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def train(self, epochs=30, lr=0.1, batch_size=128):
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
            print(f"Iter {i + 1}/{epochs} - Loss: {loss_epoch / len(train_loader)}")
        self.training_time = time.time() - start_time

    def infer(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.FloatTensor(X_test).to(self.device)
        else:
            X_test = X_test.to(self.device)

        self.model.eval()
        self.likelihood.eval()
        start_time = time.time()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            self.inference_time = time.time() - start_time
            mean = pred.mean.cpu().numpy()
            std = pred.variance.sqrt().cpu().numpy()
            lower, upper = pred.confidence_region()
        return mean, std, lower.cpu().numpy(), upper.cpu().numpy()


class ExactGPModel(gpy.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpy.means.ConstantMean()
        self.covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


class MultioutputGP(gpy.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_outputs):
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size([num_outputs])
        self.mean_module = gpy.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpy.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPTrainer:
    def __init__(self, X_train, y_train, device="cpu", lr=0.003, train_iter=30, prior_noise=1e-4):
        self.lr = lr
        self.num_outputs = y_train.shape[1]
        self.train_iter = train_iter

        if device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)

        # self.X_train = torch.FloatTensor(X_train)
        # self.y_train = torch.FloatTensor(y_train)
        # Initialize noise prior, likelihood, and model

        likelihood = gpy.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        model = MultioutputGP(self.X_train, self.y_train, likelihood, self.num_outputs)

        if self.device == "cuda":
            self.model = model.cuda()
            self.likelihood = likelihood.cuda()
        else:
            self.model = model.cpu()
            self.likelihood = likelihood.cpu()

    def train(self):
        self.model.train()
        self.likelihood.train()
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpy.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        print("Training model...")
        print("Using device:", self.device)
        # train_loader = CustomDataset(self.X_train, self.y_train, batch_size=64, shuffle=True)
        # batch_size = 64
        # n_iter = len(self.X_train) // batch_size
        X = self.X_train
        y = self.y_train
        start_time = time.time()
        for i in range(self.train_iter):
            loss_epoch = 0
            # for j in range(n_iter):
            #     X = self.X_train[j * batch_size : (j + 1) * batch_size]
            #     y = self.y_train[j * batch_size : (j + 1) * batch_size]
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, self.train_iter, loss_epoch))
        self.training_time = time.time() - start_time

    def infer(self, X_test):
        test_x = X_test.to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            start_time = time.time()
            pred = self.likelihood(self.model(test_x))
            self.inference_time = time.time() - start_time
            std = pred.variance.sqrt().cpu().numpy()
            mean = pred.mean.cpu().numpy()
        return mean, std


class GPListTrainer:
    def __init__(self, X_train, y_train, lr=0.003, train_iter=30, prior_noise=1e-4):
        self.output_dim = y_train.shape[1]
        self.train_iter = train_iter
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.X_train = torch.FloatTensor(X_train)  # .to(device)
        self.y_train = torch.FloatTensor(y_train)  # .to(device)
        self.models = []
        self.likelihoods = []
        for i in range(self.output_dim):
            likelihood = gpy.likelihoods.GaussianLikelihood()
            model = ExactGPModel(self.X_train, self.y_train[:, i], likelihood)
            self.models.append(model)
            self.likelihoods.append(likelihood)
        self.model = gpy.models.IndependentModelList(*self.models)
        self.likelihood = gpy.likelihoods.LikelihoodList(*self.likelihoods)
        self.loss = gpy.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        start_time = time.time()
        self.model.train()
        self.likelihood.train()
        for i in range(self.train_iter):
            print("Iter %d/%d" % (i + 1, self.train_iter))
            self.optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -self.loss(output, self.model.train_targets)
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, self.train_iter, loss.item()))
            self.optimizer.step()
            print("End of iteration")
        self.training_time = time.time() - start_time

    def infer(self, X_test):
        test_x_all = []
        test_x = torch.FloatTensor(X_test)  # .to(self.device)
        for i in range(self.output_dim):
            test_x_all.append(test_x)
        test_x = test_x_all
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            start_time = time.time()
            predictions = self.likelihood(*self.model(*test_x))
            self.inference_time = time.time() - start_time
        means = []
        stds = []
        for prediction in predictions:
            std = prediction.stddev.detach().numpy()
            mean = prediction.mean.detach().numpy()
            means.append(mean)
            stds.append(std)
        means = np.array(means).T
        stds = np.array(stds).T
        return means, stds

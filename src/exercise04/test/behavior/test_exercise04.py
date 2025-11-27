import os
import sys
import unittest

import gpytorch as gpy
import numpy as np
import torch
from exercise04.gaussian_process import IndependentMultitaskSVGPModel as SVGPStudent
from exercise04.gaussian_process import (
    MultiOutputGaussianProcess as MultiOutputGaussianProcessStudent,
)
from exercise04.neural_network import NeuralNetwork, RegressionTrainer
from exercise04.utils import CustomDataset, Normalizer, set_seed, train_test_split

module_path = sys.modules["exercise04"].__file__
import_prefix = f"{os.path.dirname(module_path)}/" if module_path is not None else ""

def init_gp(kernel="se", kernel_params= {"length_scale": 1.0, "nu": 1.5}, noise=1e-4):
    return MultiOutputGaussianProcessStudent(kernel=kernel, kernel_params=kernel_params, noise=noise)

class BaseTest(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.gp = init_gp()
        self.X1 = np.random.rand(10, 1)
        self.X2 = np.random.rand(10, 1)

class TestKernels(BaseTest):
    def kerneltest(self, kernel:str):
        self.gp.set_kernel(kernel)
        K = self.gp.kernel(self.X1, self.X2)
        self.assertEqual(
            K.shape,
            (self.X1.shape[0], self.X2.shape[0]),
            f"{kernel} kernel shape is not correct.",
        )
        if kernel in ["se", "matern", "exp"]:
            self.gp.set_kernel_params(length_scale=2.0)
            K2 = self.gp.kernel(self.X1, self.X2)
            self.assertFalse(
                np.allclose(K, K2),
                f"{kernel} kernels with different length scales should not be the same.",
            )
            
    def test_se_kernel(self):
        self.kerneltest("se")

    def test_linear_kernel(self):
        self.kerneltest("linear")

    def test_matern_kernel(self):
        for nu in [0.5, 1.5, 2.5]:
            self.gp.set_kernel_params(nu=nu)
            self.gp.set_kernel("matern")

        self.gp.set_kernel_params(nu=0.0)  # invalid nu
        try:
            self.gp.kernel(self.X1, self.X2)
        except Exception:
            pass
        else:
            self.fail("Matern kernel with invalid nu did not raise an error.")
        
class TestFit(BaseTest):
    def setUp(self):
        super().setUp()
        self.X_train = np.random.rand(20, 5)
        self.y_train = np.random.rand(20, 2)  # Multi-dimensional outputs

    def test_fit(self):
        """Test if the fit method runs without errors and sets the training data."""
        try:
            self.gp.fit(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"Fit method failed with exception: {e}")
        
        self.assertIsNotNone(self.gp.K)
        self.assertIsNotNone(self.gp.L)
        self.assertIsNotNone(self.gp.alphas)
        self.assertEqual(self.gp.K.shape, (self.X_train.shape[0], self.X_train.shape[0]))
        self.assertEqual(self.gp.L.shape, (self.X_train.shape[0], self.X_train.shape[0]))
        self.assertEqual(self.gp.alphas.shape, (self.X_train.shape[0], self.y_train.shape[1]))

class TestPredict(BaseTest):
    def setUp(self):
        super().setUp()
        self.X_train = np.random.rand(20, 5)
        self.y_train = np.random.rand(20, 2)  # Multi-dimensional outputs
        self.X_test = np.random.rand(10, 5)  
        try:
            self.gp.fit(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"Fit method failed with exception: {e}")


    def test_mean(self):
        try:
            mean = self.gp.predict(self.X_test)
        except Exception as e:
            self.fail(f"Predict method failed with exception: {e}")

        self.assertEqual(
            mean.shape,
            (self.X_test.shape[0], self.y_train.shape[1]),
            "Mean prediction shape is incorrect.",
        )

    def test_std(self):
        """Test if the standard deviation prediction values are computed correctly."""
        _, std = self.gp.predict(self.X_test, return_std=True)
        self.assertEqual(
            std.shape,
            (self.X_test.shape[0], self.y_train.shape[1]),
            "Standard deviation prediction shape is incorrect.",
        )
        self.assertTrue(np.all(std >= 0), "Standard deviation contains negative values.")


class TestInitSVGP(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.X_train = np.random.rand(20, 5)
        self.y_train = np.random.rand(20, 2)

    def test_init_SVGP(self):
        """Test if the SVGP model initializes correctly."""
        try:
            gp = SVGPStudent(X_train=self.X_train, num_tasks=self.y_train.shape[1], num_inducing_points=self.X_train.shape[0]//2)
        except Exception as e:
            self.fail(f"SVGP initialization failed with exception: {e}")

        self.assertTrue(gp.var_distribution_type == gpy.variational.CholeskyVariationalDistribution, "Variational distribution is not of the correct type.")
        self.assertTrue(gp.var_strategy_type == gpy.variational.IndependentMultitaskVariationalStrategy, "Variational strategy is not of the correct type.")
        self.assertIsNotNone(gp.mean_module, "Mean module is not initialized.")
        self.assertIsNotNone(gp.covar_module, "Covariance module is not initialized.")

class TestCustomDatasetInit(unittest.TestCase):
    def setUp(self):
        set_seed()
        self.X = torch.randn(10, 5)
        self.y = torch.randn(10, 1)
        self.batch_size = 1

    def test_iter_shuffle(self):
        """Test if the CustomDataset can be iterated over with shuffling."""
        dataset = CustomDataset(
            self.X, self.y, batch_size=self.batch_size, shuffle=True
        )
        batches = list(iter(dataset))
        self.assertEqual(len(batches), len(dataset))
        for X_batch, y_batch in batches:
            self.assertEqual(X_batch.shape[0], y_batch.shape[0])
            self.assertLessEqual(X_batch.shape[0], self.batch_size)

    def test_iter_no_shuffle(self):
        """Test if the dataset can be iterated over without shuffling."""
        dataset = CustomDataset(
            self.X, self.y, batch_size=self.batch_size, shuffle=False
        )
        batches = list(iter(dataset))
        self.assertEqual(len(batches), len(dataset))
        for X_batch, y_batch in batches:
            self.assertEqual(X_batch.shape[0], y_batch.shape[0])
            self.assertLessEqual(X_batch.shape[0], self.batch_size)


class TestNeuralNetworkInit(unittest.TestCase):
    def setUp(self):
        set_seed()
        output_path = os.path.join(import_prefix, "outputs")
        self.checkpoint_path = os.path.join(output_path, "nn_checkpoint_ex04.ckpt")

    def test_init_NN(self):
        """Test if the NeuralNetwork class can be initialized with a checkpoint."""
        try:
            model = NeuralNetwork(
                init_from_checkpoint=True, checkpoint_path=self.checkpoint_path
            )
        except Exception as e:
            self.fail(
                f"Initialization of the Neural Network class failed with exception: {e}"
            )
        self.assertIsInstance(
            model, NeuralNetwork, "Model is not an instance of NeuralNetwork"
        )

    def test_input_output_dim_forward_pass(self):
        """Test if the forward pass works and if the output dimension matches the hyperparameters."""
        model = NeuralNetwork(
            init_from_checkpoint=True, checkpoint_path=self.checkpoint_path
        )
        batch_size = 8
        input_tensor = torch.randn(batch_size, model.hyperparameters["input_dim"])
        try:
            output_tensor = model(input_tensor)
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")
        self.assertEqual(
            output_tensor.shape,
            (batch_size, model.hyperparameters["output_dim"]),
            "Output dimension is incorrect",
        )


class TestTrainEpoch(unittest.TestCase):
    def setUp(self):
        set_seed()
        checkpoint_path = os.path.join(
            import_prefix, "outputs", "nn_checkpoint_ex04.ckpt"
        )
        self.model = NeuralNetwork(
            init_from_checkpoint=True, checkpoint_path=checkpoint_path
        )
        # Learn a simple linear relationship
        coefficients = torch.randn(
            self.model.hyperparameters["input_dim"],
            self.model.hyperparameters["output_dim"],
        )
        X = torch.randn(128, self.model.hyperparameters["input_dim"])
        y = X @ coefficients
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        normalizer = Normalizer()
        normalizer.fit(x_train, y_train)
        self.x_train, self.y_train = normalizer.transform(x_train, y_train)
        self.x_test, self.y_test = normalizer.transform(x_test, y_test)
        self.train_loader = CustomDataset(
            self.x_train, self.y_train, batch_size=8, shuffle=True
        )
        self.trainer = RegressionTrainer(self.model)

    def test_optimizer_scheduler_criterion(self):
        """Test if the optimizer, scheduler, and criterion are set correctly."""
        self.assertIsNotNone(self.trainer.optimizer, "Optimizer is not set.")
        self.assertIsNotNone(self.trainer.criterion, "Criteration is not set.")

        self.assertIsInstance(
            self.trainer.optimizer,
            torch.optim.Optimizer,
            "Optimizer is not an instance of torch.optim.Optimizer",
        )
        self.assertIsInstance(
            self.trainer.criterion,
            torch.nn.modules.loss._Loss,
            "Criterion is not an instance of torch.nn.modules.loss._Loss",
        )

        self.assertTrue(self.trainer.scheduler is None or
            isinstance(self.trainer.scheduler,  torch.optim.lr_scheduler.LRScheduler),
            "Scheduler is not an instance of torch.optim.lr_scheduler._LRScheduler or None",
        )

    def test_train_epoch(self):
        """Test if the train_epoch method updates the model parameters and if the training loss decreases."""
        initial_params = [param.clone() for param in self.model.parameters()]
        loss = self.trainer.train_epoch(self.train_loader)
        updated_params = [param for param in self.model.parameters()]
        self.assertIsInstance(loss, float, "Loss is not a float.")
        self.assertGreater(loss, 0, "Loss is negative or zero.")

        for initial, updated in zip(initial_params, updated_params):
            self.assertFalse(
                torch.equal(initial, updated), "Model parameters did not update."
            )
        print(f"Loss at the start of the training: {loss:.3g}")
        for _ in range(20):
            current_loss = self.trainer.train_epoch(self.train_loader)
        print(f"Loss after 10 epochs: {current_loss:.3g}")
        self.assertLess(
            current_loss, loss, "Loss did not decrease over first 20 epochs"
        )
        self.assertLess(current_loss, 0.1, "Loss is too high after 20 epochs")

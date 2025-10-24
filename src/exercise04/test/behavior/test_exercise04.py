import os
import sys
import unittest

import numpy as np
import torch
from exercise04.gaussian_process import (
    MultiOutputGaussianProcess as MultiOutputGaussianProcessStudent,
)
from exercise04.neural_network import NeuralNetwork, RegressionTrainer
from exercise04.utils import CustomDataset, Normalizer, set_seed, train_test_split

module_path = sys.modules["exercise04"].__file__
import_prefix = f"{os.path.dirname(module_path)}/" if module_path is not None else ""


class TestRBFKernel(unittest.TestCase):
    def setUp(self):
        # Set a random seed for reproducibility
        set_seed(42)
        self.L1 = 1.0
        self.L2 = 2.0
        self.gp = MultiOutputGaussianProcessStudent(length_scale=self.L1, noise=1e-4)
        self.gp2 = MultiOutputGaussianProcessStudent(length_scale=self.L2, noise=1e-4)
        self.X1 = np.random.rand(10, 1)
        self.X2 = np.random.rand(10, 1)

    def test_rbf_kernel_shape(self):
        rbf_kernel = self.gp.rbf_kernel(self.X1, self.X2)
        self.assertEqual(
            rbf_kernel.shape,
            (self.X1.shape[0], self.X2.shape[0]),
            "RBF kernel shape is not correct.",
        )


class TestPredict(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.gp = MultiOutputGaussianProcessStudent(length_scale=1.0, noise=1e-4)
        self.X_train = np.random.rand(20, 2)
        self.y_train = np.random.rand(20, 3)  # Multi-dimensional outputs
        self.X_test = np.random.rand(5, 2)
        self.gp.fit(self.X_train, self.y_train)

    def test_mean_shape(self):
        """Test if the mean prediction has the correct shape."""
        mean = self.gp.predict(self.X_test)
        self.assertEqual(
            mean.shape,
            (self.X_test.shape[0], self.y_train.shape[1]),
            "Mean prediction shape is incorrect.",
        )

    def test_std_shape(self):
        """Test if the standard deviation prediction values are computed correctly."""
        mean, std = self.gp.predict(self.X_test, return_std=True)
        self.assertEqual(
            std.shape,
            (self.X_test.shape[0], self.y_train.shape[1]),
            "Standard deviation prediction shape is incorrect.",
        )
        self.assertIsNotNone(std, "Standard deviation prediction is None.")


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

    def test_init(self):
        """Test if the NeuralNetwork class can be initialized with a checkpoint."""
        try:
            model = NeuralNetwork(
                init_from_checkpoint=True, checkpoint_path=self.checkpoint_path
            )
        except Exception as e:
            self.fail(
                f"Initialization of the Neural Netwrok class failed with exception: {e}"
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
        for _ in range(15):
            current_loss = self.trainer.train_epoch(self.train_loader)
        print(f"Loss after 10 epochs: {current_loss:.3g}")
        self.assertLess(
            current_loss, loss, "Loss did not decrease over first 10 epochs"
        )
        self.assertLess(current_loss, 0.1, "Loss is too high after 10 epochs")

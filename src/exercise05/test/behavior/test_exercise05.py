import os
import sys
import unittest
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch
from crazyflow.sim.symbolic import symbolic_attitude
from exercise05.gp import GaussianProcess, compute_loss_gp, fit_gp
from exercise05.gpmpc import GPMPC
from exercise05.run_gp_mpc import learn
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy

module_path = sys.modules["exercise05"].__file__
import_prefix = f"{os.path.dirname(module_path)}/" if module_path is not None else ""


def setup_gpmpc(traj=None):
    """Set up the GPMPC object and mock Gaussian Processes."""
    nominal_model_params = {
        "a": 12.1432,
        "b": 1.8118,
    }
    nominal_model = symbolic_attitude(dt=0.02, params=nominal_model_params)
    if traj is None:
        traj = np.random.rand(nominal_model.nx, 100)
    gpmpc = GPMPC(
        symbolic_model=nominal_model,
        traj=traj,
        prior_params=nominal_model_params,
        horizon=3,
        q_mpc=[8, 0.1, 8, 0.1, 8, 0.1, 0.5, 0.5, 0.5, 0.001, 0.001, 0.001],
        r_mpc=[3, 3, 3, 0.1],
        sparse_gp=True,
        prob=0.95,
        max_gp_samples=3,
        seed=1,
        device="cpu",
        output_dir=Path("/tmp"),
    )
    return gpmpc


class TestFitGP(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(100, 5)  # 100 samples, 5 features
        self.y = (self.x @ torch.rand(5, 1)).squeeze()  # Simulate a linear relationship
        self.gp = GaussianProcess(self.x, self.y)

    def test_fit_gp_linear(self):
        # Compute initial MSE loss using the mean of the MultivariateNormal
        initial_loss = compute_loss_gp(self.gp, self.x, self.y)
        # Fit the GP
        fit_gp(self.gp, n_train=50, lr=0.05, device="cpu", patience=5, rtol=1e-3)
        # Compute final MSE loss
        final_loss = compute_loss_gp(self.gp, self.x, self.y)
        # Assert that the loss has decreased by at least 50%
        self.assertLess(
            final_loss,
            initial_loss * 0.5,
            "GP fitting did not decrease the loss by at least 50%.",
        )


class TestGPMPC(unittest.TestCase):
    def setUp(self):
        self.gpmpc = setup_gpmpc()

    def test_train_gp_sample_data(self):
        # Mock data
        x = np.random.rand(10, 7)
        y = np.random.rand(10, 3)

        # Call train_gp
        self.gpmpc.train_gp(x, y, lr=0.01, iterations=1, val_split=0.0)

        # Test train_gp method
        self.assertEqual(len(self.gpmpc.gaussian_process), len(self.gpmpc.gp_idx))
        for i, gp in enumerate(self.gpmpc.gaussian_process):
            self.assertIsInstance(gp, GaussianProcess)
            self.assertTrue(gp.K.shape == torch.Size([x.shape[0], x.shape[0]]))
            self.assertTrue(gp.K_inv.shape == torch.Size([x.shape[0], x.shape[0]]))
            self.assertTrue(
                gp.train_inputs[0].shape
                == torch.Size([x.shape[0], len(self.gpmpc.gp_idx[i])])
            )
        # Test precompute_posterior_mean
        posterior_mean_numpy, train_data_numpy = self.gpmpc.precompute_posterior_mean()
        self.assertEqual(
            posterior_mean_numpy.shape[0], len(self.gpmpc.gaussian_process)
        )
        self.assertTrue(
            np.allclose(
                posterior_mean_numpy[0],
                torch.linalg.solve(
                    self.gpmpc.gaussian_process[0].K,
                    self.gpmpc.gaussian_process[0].train_targets,
                ).numpy(force=True),
            )
        )
        self.assertTrue(np.allclose(train_data_numpy[0], x[self.gpmpc.gp_idx[0], :]))

        # Test propagate_constraint_limits
        # Mock previous rollout data
        self.gpmpc.x_prev = np.random.randn(self.gpmpc.model.nx, self.gpmpc.T + 1)
        self.gpmpc.u_prev = np.random.randn(self.gpmpc.model.nu, self.gpmpc.T)
        # Call propagate_constraint_limits
        state_constraint, input_constraint = self.gpmpc.propagate_constraint_limits()

        self.assertEqual(
            state_constraint.shape,
            (self.gpmpc.model.nx * 2, self.gpmpc.T + 1),
        )
        self.assertEqual(
            input_constraint.shape,
            (self.gpmpc.model.nu * 2, self.gpmpc.T),
        )


class TestLearnFunction(unittest.TestCase):
    def setUp(self):
        # Set up the environment and controller for testing
        self.env = JaxToNumpy(gymnasium.make_vec("DroneFigureEightTrajectory-v0", num_envs=1))
        # self.traj = self.env.unwrapped.trajectory.T
        self.ctrl = setup_gpmpc(traj=self.env.unwrapped.trajectory.T)

    @pytest.mark.timeout(90)
    def test_learn_shapes(self):
        # Define parameters for the learn function

        n_epochs = 1
        max_samples = 2

        # Call the learn function
        train_runs, test_runs = learn(
            n_epochs=n_epochs,
            ctrl=self.ctrl,
            env=self.env,
            lr=1e-3,
            gp_iterations=2,
            seed=1,
            max_samples=max_samples,
        )

        for gp in self.ctrl.gaussian_process:
            self.assertIsInstance(gp, GaussianProcess)
            self.assertTrue(gp.K.shape == torch.Size([max_samples, max_samples]))
            self.assertTrue(gp.n_ind_points == max_samples)
            # print("train inputs", gp.train_inputs[0].shape)
            # print("train targets", gp.train_targets.shape)
            # print("num inducing points", gp.n_ind_points)
            # print("K shape", gp.K.shape)
            # print("K_inv shape", gp.K_inv.shape)

        # Check that train_runs and test_runs have the correct number of epochs
        self.assertEqual(len(train_runs), n_epochs + 1)  # Includes the prior run
        self.assertEqual(len(test_runs), n_epochs + 1)

        # Check that the training and testing data have the correct shapes
        for epoch in range(n_epochs + 1):
            self.assertIn("obs", train_runs[epoch])
            self.assertIn("action", train_runs[epoch])
            self.assertIn("obs", test_runs[epoch])
            self.assertIn("action", test_runs[epoch])


if __name__ == "__main__":
    unittest.main()

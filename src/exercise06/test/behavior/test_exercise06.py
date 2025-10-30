import os
import pickle
import random
import sys
import unittest
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch
from exercise06.ppo import PPOTester, PPOTrainer, make_envs, save_model, set_seeds
from exercise06.rand_traj_env import RandTrajEnv
from exercise06.wrappers import AngleReward, FlattenJaxObservation

sys.path.append(
    os.path.abspath(os.path.dirname(__file__))
)  # make sure utils_test is found in exercise and solution repo

try:
    from utils_test import compare_dicts
except ModuleNotFoundError:
    from src.utils_test import (
        compare_dicts,  # required for importing utils_test in development repository
    )

module_path = sys.modules["exercise06"].__file__
import_prefix = f"{os.path.dirname(module_path)}/" if module_path is not None else ""

RTOL = 1e-3
ATOL = 1e-4

def load_test_data():
    """Load precomputed test data from a pickle file."""
    filepath = os.path.join(
        Path(__file__).parent,
        "exercise06_testdata.pkl",
    )
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


class TestSetSeed(unittest.TestCase):
    def setUp(self):
        pass

    # this test is not perfect, as by chance the same random numbers can be generated without actually setting the seed
    def test_seeding(self):
        seed = 42
        set_seeds(seed)

        assert random.randint(0, 9999) == 1824, (
            "Random seed likely not set correctly for random module"
        )
        assert np.random.randint(0, 9999) == 7270, (
            "Random seed likely not set correctly for numpy module"
        )
        assert torch.initial_seed() == 42, (
            "Random seed not set correctly for torch module"
        )
        if torch.cuda.is_available():
            assert torch.cuda.initial_seed() == 42, (
                "Random seed not set correctly for torch.cuda module"
            )
        assert torch.backends.cudnn.deterministic is True, (
            "CUDNN not set to deterministic mode"
        )
        assert torch.backends.cudnn.benchmark is False, (
            "CUDNN benchmark mode should be disabled"
        )


class TestMakeEnvs(unittest.TestCase):
    def test_make_envs(self):
        """Test that make_envs() constructs environments with correct wrapper hierarchy."""
        train_envs, eval_envs = make_envs("DroneFigureEightTrajectory-v0", 2, 2, "cpu")

        # Check top-level wrapper
        self.assertIsInstance(
            train_envs, gymnasium.wrappers.vector.JaxToTorch, 
            "train_envs should be wrapped with JaxToTorch"
        )
        self.assertIsInstance(
            eval_envs, gymnasium.wrappers.vector.JaxToTorch, 
            "eval_envs should be wrapped with JaxToTorch"
        )

        # Unwrap JaxToTorch
        train_envs_lvl1 = train_envs.env
        eval_envs_lvl1 = eval_envs.env

        # Check FlattenJaxObservation
        self.assertIsInstance(
            train_envs_lvl1, FlattenJaxObservation,
            "train_envs inner env should be FlattenJaxObservation"
        )
        self.assertIsInstance(
            eval_envs_lvl1, FlattenJaxObservation,
            "eval_envs inner env should be FlattenJaxObservation"
        )

        # Unwrap FlattenJaxObservation
        train_envs_lvl2 = train_envs_lvl1.env
        eval_envs_lvl2 = eval_envs_lvl1.env

        # Check AngleReward
        self.assertIsInstance(
            train_envs_lvl2, AngleReward,
            "train_envs inner env should be AngleReward"
        )
        self.assertIsInstance(
            eval_envs_lvl2, AngleReward,
            "eval_envs inner env should be AngleReward"
        )
        

class TestSaveModel(unittest.TestCase):
    def setUp(self):
        self.test_data = load_test_data()
        seed = self.test_data["setup"]["seed"]
        set_seeds(seed)  #
        self.agent = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)
        self.train_envs, _ = make_envs("DroneFigureEightTrajectory-v0", 2, 2, "cpu")

    def test_save_model(self):
        dict_ = save_model(
            self.agent, self.optimizer, self.train_envs, None, save=False
        )
        expected_dict = self.test_data["save_model"]["outputs"]["save_dict"]
        compare_dicts(self, expected_dict, dict_)


class TestCalculateAdvantages(unittest.TestCase):
    def setUp(self):
        """Load precomputed test data."""
        self.test_data = load_test_data()
        self.trainer = PPOTrainer(
            config=self.test_data["setup"]["config"],
            wandb_log=self.test_data["setup"]["wandb_log"],
        )
        self.trainer.rewards_buffer = self.test_data["setup"]["rewards_buffer"]
        self.trainer.values_buffer = self.test_data["setup"]["values_buffer"]
        self.trainer.terminated_buffer = self.test_data["setup"]["terminated_buffer"]

    def test_advantages(self):
        """Test calculate_advantages function."""
        self.test_data = load_test_data()
        inputs = self.test_data["calculate_advantages"]["inputs"]
        _, advantages = self.trainer.calculate_advantages(**inputs)
        expected_advantages = self.test_data["calculate_advantages"]["outputs"][
            "advantages"
        ]
        #
        torch.testing.assert_close(
            advantages,
            expected_advantages,
            atol=ATOL,
            rtol=RTOL,
            msg="GAE formula incorrect: advantage calculation mismatch. Hint: Check how your`delta` is computed, and did you include 'nextnonterminal' to handle episode termination correctly?",
        )


class TestCalculatePGLoss(unittest.TestCase):
    def setUp(self):
        """Load precomputed test data."""
        self.test_data = load_test_data()
        self.trainer = PPOTrainer(
            config=self.test_data["setup"]["config"],
            wandb_log=self.test_data["setup"]["wandb_log"],
        )

    def test_pg_loss(self):
        """Test calculate_pg_loss function."""
        inputs = self.test_data["calculate_pg_loss"]["inputs"]
        pg_loss = self.trainer.calculate_pg_loss(**inputs)
        expected_pg_loss = self.test_data["calculate_pg_loss"]["outputs"]["pg_loss"]
        torch.testing.assert_close(
            pg_loss,
            expected_pg_loss,
            atol=ATOL,
            rtol=RTOL,
            msg="`pg_loss` is incorrect. Make sure to use the max operator, and make the advantages negative!",
        )


class TestCalculateVLoss(unittest.TestCase):
    def setUp(self):
        """Load precomputed test data."""
        self.test_data = load_test_data()
        self.trainer = PPOTrainer(
            config=self.test_data["setup"]["config"],
            wandb_log=self.test_data["setup"]["wandb_log"],
        )

    def test_calculate_v_loss_unclipped(self):
        """Test calculate_pg_loss function.(unclipped)."""
        inputs = self.test_data["calculate_v_loss"]["inputs"]
        v_loss_unclipped = self.trainer.calculate_v_loss(**inputs, if_clip=False)
        expected_v_loss_unclipped = self.test_data["calculate_v_loss"]["outputs"][
            "v_loss_unclipped"
        ]
        #
        torch.testing.assert_close(
            v_loss_unclipped,
            expected_v_loss_unclipped,
            atol=ATOL,
            rtol=RTOL,
            msg=f"Unclipped v_loss incorrect. Expected: {expected_v_loss_unclipped}, Got: {v_loss_unclipped}",
        )

    def test_calculate_v_loss_clipped(self):
        """Test calculate_pg_loss function (clipped)."""
        inputs = self.test_data["calculate_v_loss"]["inputs"]
        v_loss_clipped = self.trainer.calculate_v_loss(**inputs, if_clip=True)
        expected_v_loss_clipped = self.test_data["calculate_v_loss"]["outputs"][
            "v_loss_clipped"
        ]
        #
        torch.testing.assert_close(
            v_loss_clipped,
            expected_v_loss_clipped,
            atol=ATOL,
            rtol=RTOL,
            msg=f"Clipped v_loss incorrect. Expected: {expected_v_loss_clipped}, Got: {v_loss_clipped}",
        )

class TestRandTrajEnv(unittest.TestCase):
    def setUp(self):
        """Create RandTrajEnv instance."""
        self.student_env = RandTrajEnv()

    def test_trajectory_attr_existance(self):
        """Test whether RandTrajEnv has either 'trajectory' or 'trajectories' attribute."""
        # Check if either 'trajectory' or 'trajectories' attribute exists
        has_trajectory = hasattr(self.student_env, "trajectory")
        has_trajectories = hasattr(self.student_env, "trajectories")

        # At least one of them must exist
        self.assertTrue(
            has_trajectory or has_trajectories,
            "RandTrajEnv must have at least one attribute: 'trajectory' or 'trajectories'.",
        )

class TestFinalPolicyPublic(unittest.TestCase):
    def setUp(self):
        """Load precomputed test data."""
        self.test_data = load_test_data()
        # Load saved sample trajectory
        self.example_trajectory = self.test_data["random_trajectories"]["example_trajectory"]

    @pytest.mark.timeout(80)
    def test_policy_reach_pos(self):
        env_name="DroneReachPos-v0"
        ckpt_path = import_prefix + f"ppo_checkpoint_{env_name}.pt"
        reward, _ = PPOTester(seed=42, ckpt_path=ckpt_path, n_episodes=5, env_name=env_name)
        # Check that reward for ReachPosEnv is over 380.
        self.assertGreater(
            reward,
            380,
            f"Your policy received a reward of {reward}, which is not good enough. You require at least 380 reward. The PPO learning parameters we provide should suffice to achieve that reward. However, you can still tweak them. Most likely though, something is wrong with your implementation!",
        )

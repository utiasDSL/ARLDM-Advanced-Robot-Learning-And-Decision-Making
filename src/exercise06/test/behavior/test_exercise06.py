import os
import pickle
import random
import sys
import unittest
from pathlib import Path

import gymnasium
import numpy as np
import torch
from exercise06.ppo import PPOTrainer, make_envs, save_model, set_seeds

sys.path.append(
    os.path.abspath(os.path.dirname(__file__))
)  # make sure utils_test is found in exercise and solution repo

try:
    from utils_test import compare_dicts
except ModuleNotFoundError:
    from src.utils_test import (
        compare_dicts,  # required for importing utils_test in development repository
    )

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
        train_envs, eval_envs = make_envs(2, 2, "cpu")
        assert isinstance(
            train_envs, gymnasium.wrappers.vector.JaxToTorch
        ), "train_envs is not wrapped correctly"

        assert isinstance(
            eval_envs, gymnasium.wrappers.vector.JaxToTorch
        ), "eval_envs is not wrapped correctly"
        

class TestSaveModel(unittest.TestCase):
    def setUp(self):
        self.test_data = load_test_data()
        seed = self.test_data["setup"]["seed"]
        set_seeds(seed)  #
        self.agent = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)
        self.train_envs, _ = make_envs(2, 2, "cpu")

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

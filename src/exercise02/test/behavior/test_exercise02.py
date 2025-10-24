import os
import pickle
import unittest
from pathlib import Path

import gymnasium
import numpy as np
from exercise02.ilqr import ILQR
from exercise02.lqr import LQR
from exercise02.utils import discretize_linear_system, obs_to_state
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation as R  #

env = gymnasium.make_vec(
    "DroneReachPos-v0",
    num_envs=1,
    freq=500,
    device="cpu",
)


def load_test_data():
    """Load precomputed test data from a pickle file."""
    filepath = os.path.join(
        Path(__file__).parent,
        "exercise02_testdata.pkl",
    )
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


class TestObsToState(unittest.TestCase):
    """Note: The choice of the values in the test is arbitrary and does not reflect any practical system requirement."""

    def setUp(self):
        self.test_data = load_test_data()
        self.obs_dict = self.test_data["obs_to_state"]["inputs"]["obs"]
        self.state = obs_to_state(self.obs_dict)
        self.state_expected = self.test_data["obs_to_state"]["outputs"]["state_expected"]

    def test_state_shape(self):
        self.assertEqual(
            self.state.shape,
            self.state_expected.shape,
            f"Incorrect state shape: expected {self.state_expected.shape}, got {self.state.shape}",
        )

    def test_mapping(self):
        self.assertTrue(
            np.allclose(self.state[0:6], self.state_expected[0:6]),
            "Position and euler angles are incorrectly mapped.",
        )
        

    def test_euler_angle(self):
        euler = self.state[3:6]
        quat_reconstructed = R.from_euler("xyz", euler).as_quat()
        try:
            self.assertTrue(
                np.allclose(quat_reconstructed, self.obs_dict["quat"]),
                "Error in quaternion-to-Euler conversion: incorrect Euler angles.",
            )
        except AssertionError as e:
            self.assertFalse(
                np.allclose(self.state[[3, 4, 5]], self.state_expected[[5, 4, 3]]),
                "Incorrect Euler angle order: The Euler angles appear to be reversed.",
            )
            raise e


class TestDiscretizeLinearSystem(unittest.TestCase):
    """Note: The choice of the values in the test is arbitrary and does not reflect any practical system requirement."""

    def setUp(self):
        self.test_data = load_test_data()
        inputs = self.test_data["discretize_linear_system"]["inputs"]
        self.Ad, self.Bd = discretize_linear_system(**inputs)
        self.Ad_expected = self.test_data["discretize_linear_system"]["outputs"]["Ad_expected"]
        self.Bd_expected = self.test_data["discretize_linear_system"]["outputs"]["Bd_expected"]

    def test_discretized_A(self):
        assert_array_almost_equal(
            self.Ad, self.Ad_expected, err_msg="Discretied A matrix is incorrect."
        )

    def test_discretized_B(self):
        assert_array_almost_equal(
            self.Bd, self.Bd_expected, err_msg="Discretied B matrix is incorrect."
        )


class TestComputeLQRGain(unittest.TestCase):
    def setUp(self):
        self.test_data = load_test_data()
        lqr_setup = self.test_data["lqr_setup"]
        self.lqr_controller = LQR(env=env, **lqr_setup)
        self.gain_expected = self.test_data["compute_lqr_gain"]["outputs"]["gain_expected"]

    def test_type_of_gain(self):
        gain = self.lqr_controller.gain
        type_exp = type(self.gain_expected)
        self.assertIsInstance(
            gain, type_exp, f"gain of lqr is not of type {type_exp}"
        )
        
    def test_shape_of_gain(self):
        gain = self.lqr_controller.gain
        shape_exp = self.gain_expected.shape
        assert gain.shape == shape_exp, (
            f"Shape of gain: expected {shape_exp}, got {gain.shape}"
        )
        # assert_array_almost_equal(gain, self.gain_expected, err_msg='The gain was miscalculated.')


class TestLQRStepControl(unittest.TestCase):
    def setUp(self):
        self.test_data = load_test_data()
        self.lqr_setup = self.test_data["lqr_setup"]
        self.lqr_controller = LQR(env=env, **self.lqr_setup)
        inputs = self.test_data["step_control_lqr"]["inputs"]
        self.current_state = inputs["obs"]
        self.goal = inputs["goal"]
        self.control_input_exp = self.test_data["step_control_lqr"]["outputs"]["control_input_expected"]

    def test_control_input_type(self):
        control_input = self.lqr_controller.step_control(self.current_state, self.goal)
        type_exp = type(self.control_input_exp)
        dtype_exp = self.control_input_exp.dtype
        self.assertIsInstance(
            control_input, type_exp, f"control_input is not {type_exp}"
        )
        self.assertEqual(
            control_input.dtype, dtype_exp, f"control_input is not of type {dtype_exp}"
        )

    def test_control_input_shape(self):
        control_input = self.lqr_controller.step_control(self.current_state, self.goal)
        shape_exp = self.control_input_exp.shape
        assert control_input.shape == shape_exp, (
            f"Shape of control input: expected {shape_exp}, got {control_input.shape}"
        )

    def test_control_input_clip(self):
        control_input = self.lqr_controller.step_control(self.current_state, self.goal)
        collective_thrust_expected = self.control_input_exp[0,0]
        self.assertLessEqual(
            control_input[0, 0],
            collective_thrust_expected,
            "Clipping was not applied: The control input is incorrect and appears to be outside the expected clipped range.",
        )

    def test_control_input_at_goal(self):
        control_input = self.lqr_controller.step_control(self.goal, self.goal)
        u_op = self.lqr_setup["u_eq"]
        assert_array_almost_equal(
            control_input[0, 0],
            u_op[0],
            err_msg="Control input is incorrect: Don't forget to add u_eq.",
        )


class TestILQR(unittest.TestCase):
    def setUp(self):
        self.test_data = load_test_data()
        ilqr_setup = self.test_data["ilqr_setup"]

        self.ilqr_controller = ILQR(env=env, **ilqr_setup)

    def test_dynamic_lin_disc_return_shapes(self):
        inputs = self.test_data["dynamic_lin_disc"]["inputs"]
        outputs = self.test_data["dynamic_lin_disc"]["outputs"]
        x_op = inputs["x_e"]
        u_op = inputs["u_e"]
        Ad, Bd = self.ilqr_controller.dynamic_lin_disc(x_op, u_op)
        Ad_e, Bd_e = outputs["Ad_expected"], outputs["Bd_expected"]
        self.assertIsInstance(Ad, type(Ad_e), f"Ad should be {type(Ad_e)}")
        self.assertIsInstance(Bd, type(Bd_e), f"Bd should be {type(Bd_e)}")

        expected_Ad_shape = Ad_e.shape
        expected_Bd_shape = Bd_e.shape

        self.assertEqual(
            Ad.shape,
            expected_Ad_shape,
            f"Incorrect shape for Ad: expected {expected_Ad_shape}, got {Ad.shape}",
        )
        self.assertEqual(
            Bd.shape,
            expected_Bd_shape,
            f"Incorrect shape for Bd: expected {expected_Bd_shape}, got {Bd.shape}",
        )

    def test_terminal_cost_quad_output(self):
        """Test if terminal_cost_quad returns the correct types and shapes."""
        x_N = np.zeros((12,))

        q_N, Qv_N, Qm_N = self.ilqr_controller.terminal_cost_quad(x_N)

        # type
        self.assertIsInstance(q_N, np.ndarray, "q_N should be a NumPy array")
        self.assertIsInstance(Qv_N, np.ndarray, "Qv_N should be a NumPy array")
        self.assertIsInstance(Qm_N, np.ndarray, "Qm_N should be a NumPy array")

        # shape
        self.assertEqual(
            q_N.shape, (1, 1), f"q_N shape incorrect: expected (1,1), got {q_N.shape}"
        )
        self.assertEqual(
            Qv_N.shape,
            (12, 1),
            f"Qv_N shape incorrect: expected (12,1), got {Qv_N.shape}",
        )
        self.assertEqual(
            Qm_N.shape,
            (12, 12),
            f"Qm_N shape incorrect: expected (12,12), got {Qm_N.shape}",
        )

    def test_stage_cost_quad_output(self):
        x = np.zeros((12,))
        u = np.zeros((4,))

        q, Qv, Qm, Rv, Rm, Pm = self.ilqr_controller.stage_cost_quad(x, u)

        # type
        self.assertIsInstance(q, np.ndarray, "q should be a NumPy array")
        self.assertIsInstance(Qv, np.ndarray, "Qv should be a NumPy array")
        self.assertIsInstance(Qm, np.ndarray, "Qm should be a NumPy array")
        self.assertIsInstance(Rv, np.ndarray, "Rv should be a NumPy array")
        self.assertIsInstance(Rm, np.ndarray, "Rm should be a NumPy array")
        self.assertIsInstance(Pm, np.ndarray, "Pm should be a NumPy array")

        # shape
        self.assertEqual(
            q.shape, (1, 1), f"q shape incorrect: expected (1,1), got {q.shape}"
        )
        self.assertEqual(
            Qv.shape, (12, 1), f"Qv shape incorrect: expected (12,1), got {Qv.shape}"
        )
        self.assertEqual(
            Qm.shape, (12, 12), f"Qm shape incorrect: expected (12,12), got {Qm.shape}"
        )
        self.assertEqual(
            Rv.shape, (4, 1), f"Rv shape incorrect: expected (4,1), got {Rv.shape}"
        )
        self.assertEqual(
            Rm.shape, (4, 4), f"Rm shape incorrect: expected (4,4), got {Rm.shape}"
        )
        self.assertEqual(
            Pm.shape, (4, 12), f"Pm shape incorrect: expected (4,12), got {Pm.shape}"
        )

    def test_update_policy_return_shape(self):
        np.random.seed(42)
        Ad_k = np.random.randn(12, 12)
        Bd_k = np.random.randn(12, 4)
        q = np.random.randn(1, 1)
        Qv = np.random.randn(12, 1)
        Qm = np.random.randn(12, 12)
        Rv = np.random.randn(4, 1)
        Rm = np.random.randn(4, 4)
        Pm = np.random.randn(4, 12)
        s_next = np.random.randn(1, 1)
        Sv_next = np.random.randn(12, 1)
        Sm_next = np.random.randn(12, 12)
        x_curr = np.random.randn(12)
        u_curr = np.random.randn(4)

        theta_ff, theta_fb, s, Sv, Sm = self.ilqr_controller.update_policy(
            Ad_k, Bd_k, q, Qv, Qm, Rv, Rm, Pm, s_next, Sv_next, Sm_next, x_curr, u_curr
        )

        self.assertEqual(
            theta_ff.shape,
            (4,),
            f"q shape incorrect: expected (4,), got {theta_ff.shape}",
        )
        self.assertEqual(
            theta_fb.shape,
            (4, 12),
            f"Qv shape incorrect: expected (4, 12), got {theta_fb.shape}",
        )
        self.assertEqual(
            s.shape, (1, 1), f"Qm shape incorrect: expected (1, 1), got {s.shape}"
        )
        self.assertEqual(
            Sv.shape, (12, 1), f"Rv shape incorrect: expected (12, 1), got {Sv.shape}"
        )
        self.assertEqual(
            Sm.shape, (12, 12), f"Rm shape incorrect: expected (12, 12), got {Sm.shape}"
        )

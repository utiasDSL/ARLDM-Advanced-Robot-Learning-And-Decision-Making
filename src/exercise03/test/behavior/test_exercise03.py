import os
import sys
import unittest
import unittest.mock

import casadi as ca
import gymnasium
import numpy as np
import yaml
from acados_template import AcadosOcp  # type: ignore
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.symbolic import symbolic_from_sim
from exercise03.linear_mpc import LinearModelPredictiveController
from exercise03.mpc_utils import (
    create_linear_prediction_model,
    create_nonlinear_prediction_model,
    create_ocp_constraints,
    create_ocp_costs_linear,
    create_ocp_costs_nonlinear,
    create_ocp_solver,
)
from exercise03.nonlinear_mpc import NonlinearModelPredictiveController
from exercise03.ocp_setup import create_ocp_linear, create_ocp_nonlinear

module_path = sys.modules["exercise03"].__file__
import_prefix = f"{os.path.dirname(module_path)}/" if module_path is not None else ""

env = gymnasium.make_vec(
    "DroneReachPos-v0",
    num_envs=1,
    freq=500,
    device="cpu",
)

symbolic_model = symbolic_from_sim(env.sim)


class TestCreateLinearPredictionModel(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/linear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ts = config["solver"]["Ts"]
        self.model = create_linear_prediction_model(symbolic_model, ts)

    def test_linear_mpc_x_type(self):
        self.assertIsInstance(self.model.x, ca.MX, "`model.x` is not of type CasADi MX")

    def test_linear_mpc_x_shape(self):
        self.assertEqual(
            self.model.x.shape, (12, 1), "`model.x` does not have shape (12,1)"
        )

    def test_linear_mpc_x_is_symbolic(self):
        """Test that x contains only symbolic variables (no mathematical operations)."""
        for i in range(self.model.x.shape[0]):
            self.assertTrue(
                self.model.x[i].is_symbolic(), f"model.x[{i}] is not purely symbolic"
            )

    def test_linear_mpc_u_type(self):
        self.assertIsInstance(self.model.u, ca.MX, "`model.u` is not of type CasADi MX")

    def test_linear_mpc_u_shape(self):
        self.assertEqual(
            self.model.u.shape, (4, 1), "`model.u` does not have shape (4,1)"
        )

    def test_linear_mpc_u_is_symbolic(self):
        """Test that u contains only symbolic variables (no mathematical operations)."""
        for i in range(self.model.u.shape[0]):
            self.assertTrue(
                self.model.u[i].is_symbolic(), f"model.u[{i}] is not purely symbolic"
            )

    def test_linear_mpc_disc_dyn_expr_type(self):
        self.assertIsInstance(
            self.model.disc_dyn_expr,
            ca.MX,
            "`model.disc_dyn_expr` is not a CasADi MX expression",
        )

    def test_linear_mpc_disc_dyn_expr_shape(self):
        self.assertEqual(
            self.model.disc_dyn_expr.shape,
            (12, 1),
            "`model.disc_dyn_expr` does not have shape (12,1)",
        )

    def test_linear_mpc_disc_dyn_expr_has_operations(self):
        """Test that f_expl_expr contains at least one mathematical operation."""
        has_operations = all(
            self.model.disc_dyn_expr[i].is_symbolic()
            for i in range(self.model.disc_dyn_expr.shape[0])
        )
        self.assertFalse(
            has_operations,
            "`model.disc_dyn_expr` does not contain mathematical operations",
        )


class TestCreateNonlinearPredictionModel(unittest.TestCase):
    def setUp(self):
        self.model = create_nonlinear_prediction_model(symbolic_model)
        self.test = np.array([[[1, 2, 3]]])

    def test_nonlinear_mpc_x_type(self):
        self.assertIsInstance(self.model.x, ca.MX, "`model.x` is not of type CasADi MX")

    def test_nonlinear_mpc_x_shape(self):
        self.assertEqual(
            self.model.x.shape, (12, 1), "`model.x` does not have shape (12,1)"
        )

    def test_nonlinear_mpc_x_is_symbolic(self):
        """Test that x contains only symbolic variables (no mathematical operations)."""
        for i in range(self.model.x.shape[0]):
            self.assertTrue(
                self.model.x[i].is_symbolic(), f"model.x[{i}] is not purely symbolic"
            )

    def test_nonlinear_mpc_u_type(self):
        self.assertIsInstance(self.model.u, ca.MX, "`model.u` is not of type CasADi MX")

    def test_nonlinear_mpc_u_shape(self):
        self.assertEqual(
            self.model.u.shape, (4, 1), "`model.u` does not have shape (4,1)"
        )

    def test_nonlinear_mpc_u_is_symbolic(self):
        """Test that u contains only symbolic variables (no mathematical operations)."""
        for i in range(self.model.u.shape[0]):
            self.assertTrue(
                self.model.u[i].is_symbolic(), f"model.u[{i}] is not purely symbolic"
            )

    def test_nonlinear_mpc_xdot_type(self):
        self.assertIsInstance(
            self.model.xdot, ca.MX, "`model.xdot` is not of type CasADi MX"
        )

    def test_nonlinear_mpc_xdot_shape(self):
        self.assertEqual(
            self.model.xdot.shape, (12, 1), "`model.xdot` does not have shape (12,1)"
        )

    def test_nonlinear_mpc_xdot_is_symbolic(self):
        """Test that xdot contains only symbolic variables (no mathematical operations)."""
        all_elements_symbolic = all(
            self.model.xdot[i].is_symbolic() for i in range(self.model.xdot.shape[0])
        )
        array_symbolic = self.model.xdot.is_symbolic()
        self.assertTrue(
            all_elements_symbolic or array_symbolic,
            "model.xdot is not symbolic element-wise nor as a whole",
        )

    def test_nonlinear_mpc_f_expl_expr_type(self):
        self.assertIsInstance(
            self.model.f_expl_expr,
            ca.MX,
            "`model.f_expl_expr` is not a CasADi MX expression",
        )

    def test_nonlinear_mpc_f_expl_expr_shape(self):
        self.assertEqual(
            self.model.f_expl_expr.shape,
            (12, 1),
            "`model.f_expl_expr` does not have shape (12,1)",
        )

    def test_nonlinear_mpc_f_expl_expr_has_operations(self):
        """Test that f_expl_expr contains at least one mathematical operation."""
        has_operations = all(
            self.model.f_expl_expr[i].is_symbolic()
            for i in range(self.model.f_expl_expr.shape[0])
        )
        self.assertFalse(
            has_operations,
            "`model.f_expl_expr` does not contain mathematical operations",
        )

    def test_nonlinear_mpc_f_impl_expr_type(self):
        self.assertIsInstance(
            self.model.f_impl_expr,
            ca.MX,
            "`model.f_impl_expr` is not a CasADi MX expression",
        )

    def test_nonlinear_mpc_f_impl_expr_shape(self):
        self.assertEqual(
            self.model.f_impl_expr.shape,
            (12, 1),
            "`model.f_impl_expr` does not have shape (12,1)",
        )

    def test_nonlinear_mpc_f_impl_expr_has_operations(self):
        """Test that f_impl_expr contains at least one mathematical operation."""
        has_operations = all(
            self.model.f_impl_expr[i].is_symbolic()
            for i in range(self.model.f_impl_expr.shape[0])
        )
        self.assertFalse(
            has_operations,
            "`model.f_impl_expr` does not contain mathematical operations",
        )

    def test_nonlinear_mpc_cost_y_expr_type(self):
        self.assertIsInstance(
            self.model.cost_y_expr,
            ca.MX,
            "`model.cost_y_expr` is not of type CasADi MX",
        )

    def test_nonlinear_mpc_cost_y_expr_shape(self):
        self.assertEqual(
            self.model.cost_y_expr.shape,
            (16, 1),
            "`model.cost_y_expr` does not have shape (16,1)",
        )

    def test_nonlinear_mpc_cost_y_expr_is_symbolic(self):
        """Test that cost_y_expr contains only symbolic variables (no mathematical operations)."""
        for i in range(self.model.cost_y_expr.shape[0]):
            self.assertTrue(
                self.model.cost_y_expr[i].is_symbolic(),
                f"model.cost_y_expr[{i}] is not purely symbolic",
            )

    def test_nonlinear_mpc_cost_y_expr_e_type(self):
        self.assertIsInstance(
            self.model.cost_y_expr_e,
            ca.MX,
            "`model.cost_y_expr_e` is not of type CasADi MX",
        )

    def test_nonlinear_mpc_cost_y_expr_e_shape(self):
        self.assertEqual(
            self.model.cost_y_expr_e.shape,
            (12, 1),
            "`model.cost_y_expr_e` does not have shape (12,1)",
        )

    def test_nonlinear_mpc_cost_y_expr_e_is_symbolic(self):
        """Test that cost_y_expr_e contains only symbolic variables (no mathematical operations)."""
        for i in range(self.model.cost_y_expr_e.shape[0]):
            self.assertTrue(
                self.model.cost_y_expr_e[i].is_symbolic(),
                f"model.cost_y_expr_e[{i}] is not purely symbolic",
            )


class TestCreateOCPConstraints(unittest.TestCase):
    def setUp(self):
        self.ocp = AcadosOcp()
        with open(import_prefix + "configs/linear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ts = config["solver"]["Ts"]
        self.ocp.model = create_linear_prediction_model(symbolic_model, ts)
        self.ocp = create_ocp_constraints(self.ocp, None)

    def test_mpc_x0_type(self):
        self.assertIsInstance(
            self.ocp.constraints.x0,
            np.ndarray,
            "`constraints.x0` is not of type `np.ndarray`",
        )

    def test_mpc_x0_shape(self):
        nx = symbolic_model.nx
        self.assertEqual(
            self.ocp.constraints.x0.shape,
            (nx,),
            f"`constraints.x0` does not have shape ({nx},)",
        )

    def test_mpc_lbu_type(self):
        self.assertIsInstance(
            self.ocp.constraints.lbu,
            np.ndarray,
            "`constraints.lbu` is not of type `np.ndarray`",
        )

    def test_mpc_lbu_shape(self):
        nu = symbolic_model.nu
        self.assertEqual(
            self.ocp.constraints.lbu.shape,
            (nu,),
            f"`constraints.lbu` does not have shape ({nu},)",
        )

    def test_mpc_ubu_type(self):
        self.assertIsInstance(
            self.ocp.constraints.ubu,
            np.ndarray,
            "`constraints.ubu` is not of type `np.ndarray`",
        )

    def test_mpc_ubu_shape(self):
        nu = symbolic_model.nu
        self.assertEqual(
            self.ocp.constraints.ubu.shape,
            (nu,),
            f"`constraints.ubu` does not have shape ({nu},)",
        )

    def test_mpc_idxbu_type(self):
        self.assertIsInstance(
            self.ocp.constraints.idxbu,
            np.ndarray,
            "`constraints.idxbu` is not of type `np.ndarray`",
        )

    def test_mpc_idxbu_shape(self):
        nu = symbolic_model.nu
        self.assertEqual(
            self.ocp.constraints.idxbu.shape,
            (nu,),
            f"`constraints.idxbu` does not have shape ({nu},)",
        )


class TestCreateOCPCostsLinear(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/linear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        options = config["cost"]
        ts = config["solver"]["Ts"]
        self.ocp = AcadosOcp()
        self.ocp.model = create_linear_prediction_model(symbolic_model, ts)
        self.ocp = create_ocp_costs_linear(self.ocp, options)
        self.nx = symbolic_model.nx
        self.nu = symbolic_model.nu
        self.n = self.nx + self.nu

    def test_linear_mpc_cost_type(self):
        self.assertEqual(
            self.ocp.cost.cost_type, "LINEAR_LS", "`cost.cost_type` isn't 'LINEAR_LS'"
        )

    def test_linear_mpc_cost_type_e(self):
        self.assertEqual(
            self.ocp.cost.cost_type_e,
            "LINEAR_LS",
            "`cost.cost_type_e` isn't 'LINEAR_LS'",
        )

    def test_linear_mpc_Vx_shape(self):
        self.assertEqual(
            self.ocp.cost.Vx.shape,
            (self.n, self.nx),
            f"`cost.Vx` does not have shape {self.n, self.nx}",
        )

    def test_linear_mpc_Vx_e_shape(self):
        self.assertEqual(
            self.ocp.cost.Vx_e.shape,
            (self.nx, self.nx),
            f"`cost.Vx_e` does not have shape {self.nx, self.nx}",
        )

    def test_linear_mpc_Vu_shape(self):
        self.assertEqual(
            self.ocp.cost.Vu.shape,
            (self.n, self.nu),
            f"`cost.Vu` does not have shape {self.n, self.nu}",
        )

    def test_linear_mpc_W_shape(self):
        self.assertEqual(
            self.ocp.cost.W.shape,
            (self.n, self.n),
            f"`cost.W` does not have shape {self.n, self.n}",
        )

    def test_linear_mpc_W_e_shape(self):
        self.assertEqual(
            self.ocp.cost.W_e.shape,
            (self.nx, self.nx),
            f"`cost.W_e` does not have shape {self.nx, self.nx}",
        )

    def test_linear_mpc_yref_shape(self):
        self.assertEqual(
            self.ocp.cost.yref.shape,
            (self.n,),
            f"`cost.yref` does not have shape {(self.n,)}",
        )

    def test_linear_mpc_yref_e_shape(self):
        self.assertEqual(
            self.ocp.cost.yref_e.shape,
            (self.nx,),
            f"`cost.yref_e` does not have shape {(self.nx,)}",
        )


class TestCreateOCPCostsNonlinear(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/nonlinear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        options = config["cost"]
        self.ocp = AcadosOcp()
        self.ocp.model = create_nonlinear_prediction_model(symbolic_model)
        self.ocp = create_ocp_costs_nonlinear(self.ocp, options)
        self.nx = symbolic_model.nx
        self.nu = symbolic_model.nu
        self.n = self.nx + self.nu

    def test_nonlinear_mpc_cost_type(self):
        self.assertEqual(
            self.ocp.cost.cost_type,
            "NONLINEAR_LS",
            "`cost.cost_type` isn't 'NONLINEAR_LS'",
        )

    def test_nonlinear_mpc_cost_type_e(self):
        self.assertEqual(
            self.ocp.cost.cost_type_e,
            "NONLINEAR_LS",
            "`cost.cost_type_e` isn't 'NONLINEAR_LS'",
        )

    def test_nonlinear_mpc_W_shape(self):
        self.assertEqual(
            self.ocp.cost.W.shape,
            (self.n, self.n),
            f"`cost.W` does not have shape {self.n, self.n}",
        )

    def test_nonlinear_mpc_W_e_shape(self):
        self.assertEqual(
            self.ocp.cost.W_e.shape,
            (self.nx, self.nx),
            f"`cost.W_e` does not have shape {self.nx, self.nx}",
        )

    def test_nonlinear_mpc_yref_shape(self):
        self.assertEqual(
            self.ocp.cost.yref.shape,
            (self.n,),
            f"`cost.yref` does not have shape {(self.n,)}",
        )

    def test_nonlinear_mpc_yref_e_shape(self):
        self.assertEqual(
            self.ocp.cost.yref_e.shape,
            (self.nx,),
            f"`cost.yref_e` does not have shape {(self.nx,)}",
        )


class TestCreateOCPSolver(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/nonlinear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.options = config["solver"]
        self.options["hessian_approx"] = "EXACT"
        self.options["qp_solver"] = "FULL_CONDENSING_HPIPM"
        self.options["nlp_solver_max_iter"] = 99
        self.options["globalization_fixed_step_length"] = 0.98
        self.ocp = AcadosOcp()
        self.ocp = create_ocp_solver(self.ocp, self.options)

    def test_mpc_N_horizon(self):
        self.assertEqual(
            self.ocp.solver_options.N_horizon,
            self.options["n_pred"],
            "`solver_options.N_horizon` should be configured via `options`.",
        )

    def test_mpc_tf(self):
        ans = self.options["Ts"] * self.options["n_pred"]
        self.assertEqual(
            self.ocp.solver_options.tf, ans, "`solver_options.tf` incorrect"
        )

    def test_mpc_nlp_solver_type(self):
        self.assertEqual(
            self.ocp.solver_options.nlp_solver_type,
            self.options["nlp_solver_type"],
            "`solver_options.nlp_solver_type` should be configured via `options`.",
        )

    def test_mpc_hessian_approx(self):
        self.assertEqual(
            self.ocp.solver_options.hessian_approx,
            self.options["hessian_approx"],
            "`solver_options.hessian_approx` should be configured via `options`.",
        )

    def test_mpc_integrator_type(self):
        self.assertEqual(
            self.ocp.solver_options.integrator_type,
            self.options["integrator_type"],
            "`solver_options.integrator_type` should be configured via `options`.",
        )

    def test_mpc_qp_solver(self):
        self.assertEqual(
            self.ocp.solver_options.qp_solver,
            self.options["qp_solver"],
            "`solver_options.qp_solver` should be configured via `options`.",
        )

    def test_mpc_nlp_solver_max_iter(self):
        self.assertEqual(
            self.ocp.solver_options.nlp_solver_max_iter,
            self.options["nlp_solver_max_iter"],
            "`solver_options.nlp_solver_max_iter` should be configured via `options`.",
        )

    def test_mpc_nlp_solver_tol_comp(self):
        self.assertEqual(
            self.ocp.solver_options.nlp_solver_tol_comp,
            self.options["nlp_solver_tol_comp"],
            "`solver_options.nlp_solver_tol_comp` should be configured via `options`.t",
        )

    def test_mpc_nlp_solver_tol_eq(self):
        self.assertEqual(
            self.ocp.solver_options.nlp_solver_tol_eq,
            self.options["nlp_solver_tol_eq"],
            "`solver_options.nlp_solver_tol_eq` should be configured via `options`.",
        )

    def test_mpc_nlp_solver_tol_stat(self):
        self.assertEqual(
            self.ocp.solver_options.nlp_solver_tol_stat,
            self.options["nlp_solver_tol_stat"],
            "`solver_options.nlp_solver_tol_stat` should be configured via `options`.",
        )

    def test_mpc_nlp_solver_tol_ineq(self):
        self.assertEqual(
            self.ocp.solver_options.nlp_solver_tol_ineq,
            self.options["nlp_solver_tol_ineq"],
            "`solver_options.nlp_solver_tol_ineq` should be configured via `options`.",
        )

    def test_mpc_globalization_fixed_step_length(self):
        self.assertEqual(
            self.ocp.solver_options.globalization_fixed_step_length,
            self.options["globalization_fixed_step_length"],
            "`solver_options.globalization_fixed_step_length` should be configured via `options`.",
        )

    def test_mpc_code_export_directory(self):
        self.assertEqual(
            self.ocp.code_export_directory,
            self.options["export_dir"],
            "`solver_options.code_export_directory` should be configured via `options`.",
        )


class TestCreateOCPLinear(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/linear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.ocp, self.ocp_solver = create_ocp_linear(symbolic_model, config)

    def test_linear_mpc_ocp_constraints(self):
        self.assertNotEqual(
            self.ocp.constraints.lbu.shape, (0,), "Constraints are not assigned."
        )

    def test_linear_mpc_ocp_cost(self):
        nx = symbolic_model.nx
        nu = symbolic_model.nu
        n = nx + nu
        self.assertEqual(self.ocp.cost.W.shape, (n, n), "Cost is not configured.")


class TestCreateOCPNonlinear(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/nonlinear_mpc_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.ocp, self.ocp_solver = create_ocp_nonlinear(symbolic_model, config)

    def test_nonlinear_mpc_ocp_constraints(self):
        self.assertNotEqual(
            self.ocp.constraints.lbu.shape, (0,), "Constraints are not assigned."
        )

    def test_nonlinear_mpc_ocp_cost(self):
        nx = symbolic_model.nx
        nu = symbolic_model.nu
        n = nx + nu
        self.assertEqual(self.ocp.cost.W.shape, (n, n), "Cost is not configured.")


class TestStepControlLinear(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/linear_mpc_config.yaml") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.mpc = LinearModelPredictiveController(env, self.config)
        self.u_pred_pre = np.copy(self.mpc.u_pred)
        self.x_pred_pre = np.copy(self.mpc.x_pred)
        self.state = np.random.rand(12)
        goal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        goal_u = np.array([MASS * GRAVITY, 0, 0, 0])
        self.y_ref = np.concatenate([goal, goal_u])
        self.y_ref = np.tile(self.y_ref, (self.mpc.N, 1))
        self.y_ref_e = goal
        self.action = self.mpc.step_control(self.state, self.y_ref, self.y_ref_e)
        self.u_pred_after = np.copy(self.mpc.u_pred)
        self.x_pred_after = np.copy(self.mpc.x_pred)
        # mock
        self.mock_solver = unittest.mock.MagicMock()
        self.mock_solver.solve.return_value = 0
        mock_u = np.array([0.5, 0.1, -0.2, 0.3], dtype=np.float32)
        mock_x = np.random.rand(12).astype(np.float32)

        def mock_get(time_index, field):
            if field == "u":
                return mock_u.copy()
            elif field == "x":
                return mock_x.copy()
            return None

        self.mock_solver.get.side_effect = mock_get
        self.mpc.ocp_solver = self.mock_solver
        _ = self.mpc.step_control(self.state, self.y_ref, self.y_ref_e)

    def test_linear_mpc_initial_state_constraints(self):
        try:
            self.mock_solver.set.assert_any_call(0, "lbx", unittest.mock.ANY)
        except AssertionError:
            raise AssertionError(
                "Have you set lower boundary of the initial state as the current state?"
            )

        try:
            self.mock_solver.set.assert_any_call(0, "ubx", unittest.mock.ANY)
        except AssertionError:
            raise AssertionError(
                "Have you set upper boundary of the initial state as the current state?"
            )

    def test_linear_mpc_solve(self):
        try:
            self.mock_solver.solve.assert_called_once()
        except AssertionError:
            raise AssertionError("Have you solve the ocp using 'ocp_solver.solve()'?")

    def test_linear_mpc_yref(self):
        try:
            self.mock_solver.set.assert_any_call(
                unittest.mock.ANY, "yref", unittest.mock.ANY
            )
        except AssertionError:
            raise AssertionError("Have you set 'yref' correctly?")

    def test_linear_mpc_warm_start(self):
        try:
            self.mock_solver.set.assert_any_call(
                unittest.mock.ANY, "u", unittest.mock.ANY
            )
        except AssertionError:
            raise AssertionError(
                "Have you set 'u' as the previous solution to use warm start"
            )

    def test_linear_mpc_extract_u(self):
        try:
            self.mock_solver.get.assert_any_call(0, "u")
        except AssertionError:
            raise AssertionError(
                "Have you retrieved the first computed control action from the solver"
            )

    def test_linear_mpc_store_predictions(self):
        self.assertEqual(
            self.u_pred_after.shape,
            self.u_pred_pre.shape,
            "Don't change the shape of 'u_pred'.",
        )
        self.assertEqual(
            self.x_pred_after.shape,
            self.x_pred_pre.shape,
            "Don't change the shape of 'x_pred'.",
        )
        self.assertNotEqual(
            np.sum(self.u_pred_after),
            np.sum(self.u_pred_pre),
            "Do you update the 'u_pred' at each time step?",
        )
        self.assertFalse(
            np.allclose(self.x_pred_after, self.x_pred_pre),
            "Do you update the 'x_pred' at each time step?",
        )

    def test_linear_mpc_action_type(self):
        print(type(self.action))
        print(self.action.shape)
        self.assertIsInstance(
            self.action, np.ndarray, "control_input is not a NumPy array"
        )
        self.assertEqual(
            self.action.dtype, np.float32, "control_input is not of type float32"
        )

    def test_linear_mpc_action_shape(self):
        self.assertEqual(
            self.action.shape,
            (1, 4),
            f"Shape of control input: expected (1,4), got {self.action.shape}",
        )


class TestStepControlNonlinear(unittest.TestCase):
    def setUp(self):
        with open(import_prefix + "configs/nonlinear_mpc_config.yaml") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.mpc = NonlinearModelPredictiveController(env, self.config)
        self.u_pred_pre = np.copy(self.mpc.u_pred)
        self.x_pred_pre = np.copy(self.mpc.x_pred)
        self.state = np.random.rand(12)
        goal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        goal_u = np.array([MASS * GRAVITY, 0, 0, 0])
        self.y_ref = np.concatenate([goal, goal_u])
        self.y_ref = np.tile(self.y_ref, (self.mpc.N, 1))
        self.y_ref_e = goal
        self.action = self.mpc.step_control(self.state, self.y_ref, self.y_ref_e)
        self.u_pred_after = np.copy(self.mpc.u_pred)
        self.x_pred_after = np.copy(self.mpc.x_pred)
        # mock
        self.mock_solver = unittest.mock.MagicMock()
        self.mock_solver.solve.return_value = 0
        mock_u = np.array([0.5, 0.1, -0.2, 0.3], dtype=np.float32)
        mock_x = np.random.rand(12).astype(np.float32)

        def mock_get(time_index, field):
            if field == "u":
                return mock_u.copy()
            elif field == "x":
                return mock_x.copy()
            return None

        self.mock_solver.get.side_effect = mock_get
        self.mpc.ocp_solver = self.mock_solver
        _ = self.mpc.step_control(self.state, self.y_ref, self.y_ref_e)

    def test_nonlinear_mpc_initial_state_constraints(self):
        try:
            self.mock_solver.set.assert_any_call(0, "lbx", unittest.mock.ANY)
        except AssertionError:
            raise AssertionError(
                "Have you set lower boundary of the initial state as the current state?"
            )

        try:
            self.mock_solver.set.assert_any_call(0, "ubx", unittest.mock.ANY)
        except AssertionError:
            raise AssertionError(
                "Have you set upper boundary of the initial state as the current state?"
            )

    def test_nonlinear_mpc_solve(self):
        try:
            self.mock_solver.solve.assert_called_once()
        except AssertionError:
            raise AssertionError("Have you solve the ocp using 'ocp_solver.solve()'?")

    def test_nonlinear_mpc_yref(self):
        try:
            self.mock_solver.set.assert_any_call(
                unittest.mock.ANY, "yref", unittest.mock.ANY
            )
        except AssertionError:
            raise AssertionError("Have you set 'yref' correctly?")

    def test_nonlinear_mpc_warm_start(self):
        try:
            self.mock_solver.set.assert_any_call(
                unittest.mock.ANY, "u", unittest.mock.ANY
            )
        except AssertionError:
            raise AssertionError(
                "Have you set 'u' as the previous solution to use warm start"
            )

    def test_nonlinear_mpc_extract_u(self):
        try:
            self.mock_solver.get.assert_any_call(0, "u")
        except AssertionError:
            raise AssertionError(
                "Have you retrieved the first computed control action from the solver"
            )

    def test_nonlinear_mpc_store_predictions(self):
        self.assertEqual(
            self.u_pred_after.shape,
            self.u_pred_pre.shape,
            "Don't change the shape of 'u_pred'.",
        )
        self.assertEqual(
            self.x_pred_after.shape,
            self.x_pred_pre.shape,
            "Don't change the shape of 'x_pred'.",
        )
        self.assertNotEqual(
            np.sum(self.u_pred_after),
            np.sum(self.u_pred_pre),
            "Do you update the 'u_pred' at each time step?",
        )
        self.assertFalse(
            np.allclose(self.x_pred_after, self.x_pred_pre),
            "Do you update the 'x_pred' at each time step?",
        )

    def test_nonlinear_mpc_action_type(self):
        print(type(self.action))
        print(self.action.shape)
        self.assertIsInstance(
            self.action, np.ndarray, "control_input is not a NumPy array"
        )
        self.assertEqual(
            self.action.dtype, np.float32, "control_input is not of type float32"
        )

    def test_nonlinear_mpc_action_shape(self):
        self.assertEqual(
            self.action.shape,
            (1, 4),
            f"Shape of control input: expected (1,4), got {self.action.shape}",
        )

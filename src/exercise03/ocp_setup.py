import os

from acados_template import AcadosOcp, AcadosOcpSolver
from crazyflow.sim.symbolic import SymbolicModel

from exercise03.mpc_utils import (
    create_linear_prediction_model,
    create_nonlinear_prediction_model,
    create_ocp_constraints,
    create_ocp_costs_linear,
    create_ocp_costs_nonlinear,
    create_ocp_solver,
)


def create_ocp_linear(symbolic_model: SymbolicModel, options: dict):
    """Constructs the complete Acados OCP for the drone system.

    The Acados OCP is composed of the following key components. Please pay attention to the options that are used to configure the OCP.
    These are also frequently used options for other OCP problems.

    Args:
        symbolic_model (SymbolicModel): The symbolic representation of the drone model.
        options (dict): Dictionary containing configuration options for the OCP.

    Returns:
        Tuple[AcadosOcp, AcadosOcpSolver]: The constructed Acados OCP and its solver.
    """
    ocp = AcadosOcp()
    options_constraint = options["constraint"]
    options_cost = options["cost"]
    options_solver = options["solver"]
    ocp.dims.N = options_solver["n_pred"]
    ########################################################################
    # Task 5
    # TODO:
    # 1. Define the OCP model
    # 2. Apply constraints to the OCP
    # 3. Define the cost function
    # 4. Configure the OCP solver
    ########################################################################
    































    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    path = options_solver["export_dir"]
    parent_dir = os.path.dirname(path)
    ocp_solver = AcadosOcpSolver(
        ocp, json_file=parent_dir + "/acados_ocp_" + ocp.model.name + ".json"
    )

    return ocp, ocp_solver


def create_ocp_nonlinear(symbolic_model: SymbolicModel, options: dict):
    """Constructs the complete Acados OCP for the drone system.

    The Acados OCP is composed of the following key components. Please pay attention to the options that are used to configure the OCP.
    These are also frequently used options for other OCP problems.

    Args:
        symbolic_model (SymbolicModel): The symbolic representation of the drone model.
        options (dict): Dictionary containing configuration options for the OCP.

    Returns:
        Tuple[AcadosOcp, AcadosOcpSolver]: The constructed Acados OCP and its solver.
    """
    ocp = AcadosOcp()
    options_constraint = options["constraint"]
    options_cost = options["cost"]
    options_solver = options["solver"]

    ########################################################################
    # Task 9
    # TODO:
    # 1. Define the OCP model
    # 2. Apply constraints to the OCP
    # 3. Define the cost function
    # 4. Configure the OCP solver
    ########################################################################
    




























    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    path = options_solver["export_dir"]
    parent_dir = os.path.dirname(path)
    ocp_solver = AcadosOcpSolver(
        ocp, json_file=parent_dir + "/acados_ocp_" + ocp.model.name + ".json"
    )

    return ocp, ocp_solver

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from crazyflow.constants import GRAVITY, MASS
from crazyflow.control.control import MAX_THRUST, MIN_THRUST
from crazyflow.sim.symbolic import SymbolicModel
from exercise03.utils import discretize_linear_system


def create_linear_prediction_model(symbolic_model: SymbolicModel, ts) -> AcadosModel:
    """Create the acados model for the drone system.

    Args:
        symbolic_model (SymbolicModel): The symbolic representation of the
                                        drone model, containing the state and control variables, as well as system dynamics.
        ts: The integration time step for discretization.

    Returns:
        AcadosModel: The Acados model used for optimal control, containing the
                     state, control input, dynamics, and cost expressions.

    """
    model = AcadosModel()
    model.name = "drone_linear"
    ########################################################################
    # Task 1
    # TODO:
    # 1. Define `model.x` and `model.u` using the symbolic model.
    # 2. Specify the operating point (`x_op`, `u_op`).
    # 3. Compute the Jacobian matrices A and B of the system dynamics
    #    at the operating point using the symbolic model.
    # 4. Discretize the continuous-time linear system using
    #    `discretize_linear_system()` to obtain the discrete dynamics.
    # 5. Assign the resulting expression to `model.disc_dyn_expr`.
    #
    # Hint:
    # - It is recommended to choose the operating point (`x_op`, `u_op`)
    #   at or near the target point to improve approximation accuracy.
    #   Please use the default goal position as operating point. The
    #   default goal position in specified later in the notebook.
    #   Have a look there.
    # - Use the symbolic model's `df_func()` to compute the linearization
    #   around the operating point.
    # - The matrices A and B describe the linearized system in continuous
    #   time; convert them to discrete time before constructing the
    #   discrete dynamics expression.
    # - `model.disc_dyn_expr` should represent the next state expression
    #   in discrete time.
    ########################################################################
    
































    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    return model


def create_nonlinear_prediction_model(symbolic_model: SymbolicModel) -> AcadosModel:
    """Create the acados model for the drone system.

    Args:
        symbolic_model (SymbolicModel): The symbolic representation of the
                                        drone model, containing the state and control variables, as well as system dynamics.

    Returns:
        AcadosModel: The Acados model used for optimal control, containing the
                     state, control input, dynamics, and cost expressions.

    """
    model = AcadosModel()
    model.name = "drone"
    ########################################################################
    # Task 7
    # TODO:
    # 1. Define model.x, model.u, model.xdot
    # 2. Assign model.f_expl_expr, model.f_impl_expr
    # 3. Assign model.cost_y_expr, model.cost_y_expr_e
    # Hint:
    # 1. Ensure the correct order of the entries of `model.x` and `model.u`
    # 2. AcadosModel.x_dot is is an abstract representation of state
    #    derivatives, while SymbolicModel.x_dot defines how these
    #    derivatives evolve based on the system’s physics and control
    #    inputs.
    ########################################################################
    

















































    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    return model


def create_ocp_constraints(ocp: AcadosOcp, options: dict) -> AcadosOcp:
    """Create the acados constraints for system.

    The constraint forms acceptable to Acados are:
    1. Bounds: lbx <= x <= ubx, lbu <= u <= ubu
    2. Linear: lg <=Cx + Du <= ug
    3. Nonlinear: ih <= y(x, u) <= uh

    Args:
        ocp (AcadosOcp): The Acados OCP instance to which constraints are
                         applied.
        options (dict): Dictionary containing configuration options.

    Returns:
        AcadosOcp: The updated Acados OCP instance with constraints applied.
    """
    ########################################################################
    # Task 2
    # TODO:
    # 1. Fix the initial state ocp.constraints.x0
    # 2. Define input bounds:
    #    - ocp.constraints.lbu (lower bound)
    #    - ocp.constraints.ubu (upper bound)
    #    - ocp.constraints.idxbu (indices of constrained inputs)
    # Hint:
    # 1. The constraint for x0 is necessary for that the ocp needs to have
    #    x0 fixed. The values are not important, as they will be updated
    #    at each iteration, they are just a placeholder.
    # 2. `MIN_THRUST` and `MAX_THRUST` represent the minimum and maximum
    #    thrust values for a single motor, respectively. The drone is
    #    equipped with four such motors.
    ########################################################################
    










































































































    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################

    return ocp


def create_ocp_costs_linear(ocp: AcadosOcp, options: dict) -> AcadosOcp:
    """Creates the cost function for the drone system in an Acados OCP.

    Args:
        ocp (AcadosOcp): The Acados OCP instance to which cost functions are
                         applied.
        options (dict): Dictionary containing cost configuration parameters
                        with 4 keys:
            - `cost_type`: cost type for stage costs, e.g. `'NONLINEAR_LS'`
            - `cost_type_e`: cost type for terminal cost, e.g. `'NONLINEAR_LS'`
            - `Q`: List of diagonal elements for the state weight matrix
            - `R`: List of diagonal elements for the input weight matrix

    Returns:
        AcadosOcp: The updated Acados OCP instance with cost function settings
                   applied.
    """
    ########################################################################
    # Task 3
    # TODO:
    # 1. Set cost type of stage and terminal cost, using `options`
    # 2. Assign stage cost weight matrix (`W`), using `options`
    # 3. Assign terminal cost weight matrix (`W_e`), using options
    # 4. Construct `Vx`, `Vu`, and `Vx_e` matrices to map state and input
    #    to the cost function.
    # 5. Initialize the reference trajectories `yref` and `yref_e` with
    #    zero vectors as placeholders.
    # Hints:
    # - `W` should be a diagonal matrix combining `Q` (for states) and `R`
    #   (for controls), while `W_e` should only use `Q`.
    # - `Vx` and `Vu` determine how the full state-input vector is mapped
    #   into the least-squares cost. Be sure to preserve dimensions.
    # - The values of `yref` and `yref_e` are just initialization values
    #   and will be updated during execution, so the content is not yet
    #   important—only the dimensions matter.
    ########################################################################
    


































    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################
    return ocp


def create_ocp_costs_nonlinear(ocp: AcadosOcp, options: dict) -> AcadosOcp:
    """Creates the cost function for the drone system in an Acados OCP.

    Args:
        ocp (AcadosOcp): The Acados OCP instance to which cost functions are
                         applied.
        options (dict): Dictionary containing cost configuration parameters
                        with 4 keys:
            - `cost_type`: cost type for stage costs, e.g. `'NONLINEAR_LS'`
            - `cost_type_e`: cost type for terminal cost, e.g. `'NONLINEAR_LS'`
            - `Q`: List of diagonal elements for the state weight matrix
            - `R`: List of diagonal elements for the input weight matrix

    Returns:
        AcadosOcp: The updated Acados OCP instance with cost function settings
                   applied.
    """
    ########################################################################
    # Task 8
    # TODO:
    # 1. Set cost type of stage and terminal cost, using `options`
    # 2. Assign stage cost weight matrix (`W`), using `options`
    # 3. Assign terminal cost weight matrix (`W_e`), using options
    # 4. Initialize reference trajectories to zero (intermediate,terminal)
    # Hints:
    # 1. It's crucial to maintain the correct dimensions when initializing
    #    reference trajectories. The numerical values themselves are
    #    unimportant—they're simply placeholders for initialization.
    ########################################################################
    



























    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################
    return ocp


def create_ocp_solver(ocp: AcadosOcp, options: dict) -> AcadosOcp:
    """Configures the Acados OCP solver for the drone system.

    Args:
        ocp (AcadosOcp): The Acados OCP instance to which solver settings are
                         applied.
        options (dict): Dictionary containing solver configuration parameters
                        with 13 keys:
            - `export_dir`: path where acados will place the generated `C` code
               when exporting the solver
            - `n_pred`: prediction horizon (time step)
            - `Ts`: sampling time [s]
            - `integrator_type`: type of Integrator
            - `nlp_solver_type`: type of NLP solver
            - `qp_solver`: type of QP solver
            - `hessian_approx`: type of hessian approximation
            - `nlp_solver_max_iter`: maximum number of iterations for the NLP
                                     solver.
            - `nlp_solver_tol_comp`: tolerance for complementarity
                                     conditions in interior-point methods.
            - `nlp_solver_tol_eq`: tolerance for equality constraints
            - `nlp_solver_tol_stat`: tolerance for the stationarity condition
            - `nlp_solver_tol_ineq`: tolerance for inequality constraints
            - `globalization_fixed_step_length`: step size in the SQP method
    Returns:
        AcadosOcp: The updated Acados OCP instance with solver settings applied.
    """
    ########################################################################
    # Task 4
    # TODO:
    # 1. Define the prediction horizon in `time step` and in `second`
    # 2. Set solver types: NLP solver, QP solver
    # 3. Configure numerical approximations: hessian approximation and the
    #    integrator
    # 4. Define NLP solver parameters
    # 5. Set code export directory
    ########################################################################
    

























    ########################################################################
    #                           END OF YOUR CODE
    ########################################################################
    return ocp

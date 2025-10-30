from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver  # type: ignore


@dataclass
class ParameterBlock:
    name: str
    sym: cs.MX
    values: np.ndarray  # shape (n_params,)
    update_fcn: Optional[Callable[[], np.ndarray]] = None


@dataclass
class ConstraintBlock:
    name: str
    expr: cs.MX 
    ctypes: List[str]  # ["initial","path","terminal"]
    allow_slack: bool = False
    l2_slack_penalty: float = 1e6
    l1_slack_penalty: float = 1e3
    max_slack: Optional[np.ndarray] = None


class OcpRegistry:
    """Unified registry for parameters & constraints with deterministic ordering.

    Constraints can safely reference parameter symbols because symbols exist at add-time.
    """

    def __init__(self, horizon):
        self.parameters: OrderedDict[str, ParameterBlock] = OrderedDict()
        self.constraints: OrderedDict[str, ConstraintBlock] = OrderedDict()
        self.horizon = horizon

    # ---- Parameters ----
    def add_param(
        self,
        name: str,
        size: int,
        sym: Optional[cs.MX] = None,
        update_fcn: Optional[Callable[[], np.ndarray]] = None,
        init: Optional[np.ndarray] = None,
        overwrite: bool = False,
    ) -> cs.MX:
        if name in self.parameters and not overwrite:
            return self.parameters[name].sym
        if init is None:
            init = np.zeros(size)
        assert init.shape == (size,), f"Init for param {name} must be shape {(size,)}, got {init.shape}"
        if sym is None:
            sym = cs.MX.sym(f"p__{name}", size, 1)
        self.parameters[name] = ParameterBlock(name=name, sym=sym, values=init.copy(), update_fcn=update_fcn)
        return sym

    def get_param_sym(self, name: str) -> cs.MX:
        return self.parameters[name].sym

    def build_parameters(self, ocp: AcadosOcp):
        if not self.parameters:
            return
        p_syms = [blk.sym for blk in self.parameters.values()]
        p_vals = [blk.values for blk in self.parameters.values()]
        ocp.model.p = cs.vertcat(*p_syms)
        ocp.parameter_values = np.concatenate(p_vals, axis=0)

    def update_params(self, solver: AcadosOcpSolver, which: Union[list[str], bool] = True):
        if not self.parameters:
            return
        assert isinstance(which, (list, str, bool)), "which must be a list of parameter names or a bool"
        if isinstance(which, bool):
            if which:
                which = list(self.parameters.keys())
            else:
                return
        if isinstance(which, str):
            which = [which]

        per_node = []
        for blk in self.parameters.values():
            if blk.update_fcn is not None and blk.name in which:
                new_vals = blk.update_fcn()
                if new_vals is None:
                    raise ValueError(f"Parameter update_fcn returned None type for {blk.name}")
                if new_vals.shape[0] != blk.values.shape[0]:
                    raise ValueError(
                        f"Parameter update_fcn returned wrong shape for {blk.name}, expected {blk.values.shape}, got {new_vals.shape}"
                    )
                blk.values = new_vals  # store latest (optional)
            vals = blk.values
            # Support time-varying (n, T+1) if provided by update_fcn
            if vals.ndim == 1 or vals.shape[1] == 1:
                tiled = np.tile(vals[None, :], (self.horizon + 1, 1))
            elif vals.ndim == 2:
                assert vals.shape[1] == self.horizon + 1, f"Time-varying param {blk.name} must have second dim T+1"
                tiled = vals.T
            else:
                raise ValueError("Parameter values must be 1D or 2D.")
            per_node.append(tiled)
        mat = np.hstack(per_node)  # (T+1, total_params)
        solver.set_flat("p", mat.flatten())

    # ---- Constraints ----
    def add_constraint(
        self,
        name: str,
        expr: cs.MX,
        ctypes: Union[str, List[str]],
        allow_slack: bool = False,
        l2_slack_penalty: float = 1e6,
        l1_slack_penalty: float = 1e3,
        max_slack: Optional[np.ndarray] = None,
        overwrite: bool = False,
    ):
        if isinstance(ctypes, str):
            ctypes = [ctypes]

        if name in self.constraints and not overwrite:
            return

        if max_slack is not None:
            if max_slack.shape[0] <= 1:
                max_slack = np.full((expr.shape[0],), max_slack.item())
            assert max_slack.shape == (expr.shape[0],), (
                f"max_slack for {name} must be shape {(expr.shape[0],)}, got {max_slack.shape}"
            )

        self.constraints[name] = ConstraintBlock(
            name=name,
            expr=expr,
            ctypes=ctypes,
            allow_slack=allow_slack,
            l2_slack_penalty=l2_slack_penalty,
            l1_slack_penalty=l1_slack_penalty,
            max_slack=max_slack,
        )

    def build_constraints(self, ocp: AcadosOcp, tol: float = 1e-4, big_neg: float = -1e8, big_pos: float = 1e8):
        nh_0 = nh = nh_e = 0
        con0, con, cone = [], [], []
        idxsh, idxsh_e = [], []
        Zl, Zu, zl, zu = [], [], [], []
        Zl_e, Zu_e, zl_e, zu_e = [], [], [], []
        max_slack_list, max_slack_list_e = [], []
        run, run_e = 0, 0
        for name, blk in self.constraints.items():
            n = int(blk.expr.shape[0])
            if n == 0:
                continue
            for ctype in blk.ctypes:
                if ctype == "initial":
                    con0.append(blk.expr)
                    nh_0 += n
                elif ctype == "path":
                    con.append(blk.expr)
                    nh += n
                elif ctype == "terminal":
                    cone.append(blk.expr)
                    nh_e += n
                else:
                    raise ValueError(f"Unknown constraint type {ctype}")
                if blk.allow_slack:
                    # L2 penalty (large)
                    l2pen = blk.l2_slack_penalty * np.ones(n)
                    # L1 penalty (small)
                    l1pen = blk.l1_slack_penalty * np.ones(n)
                    if ctype == "terminal":
                        idxsh_e.extend(range(run_e, run_e + n))
                        run_e += n
                        Zl_e.extend(l2pen)
                        Zu_e.extend(l2pen)
                        zl_e.extend(l1pen)
                        zu_e.extend(l1pen)
                        if blk.max_slack is not None:
                            max_slack_list_e.extend(blk.max_slack)
                        else:
                            max_slack_list_e.extend([big_pos] * n)
                    elif ctype == "path":
                        idxsh.extend(range(run, run + n))
                        run += n
                        Zl.extend(l2pen)
                        Zu.extend(l2pen)
                        zl.extend(l1pen)
                        zu.extend(l1pen)
                        if blk.max_slack is not None:
                            max_slack_list.extend(blk.max_slack)
                        else:
                            max_slack_list.extend([big_pos] * n)

        # Assign to ocp
        if con0:
            ocp.model.con_h_expr_0 = cs.vertcat(*con0)
            ocp.dims.nh_0 = nh_0
            ocp.constraints.uh_0 = tol * np.ones(nh_0)
            ocp.constraints.lh_0 = big_neg * np.ones(nh_0)
        if con:
            ocp.model.con_h_expr = cs.vertcat(*con)
            ocp.dims.nh = nh
            ocp.constraints.uh = tol * np.ones(nh)
            ocp.constraints.lh = big_neg * np.ones(nh)
        if cone:
            ocp.model.con_h_expr_e = cs.vertcat(*cone)
            ocp.dims.nh_e = nh_e
            ocp.constraints.uh_e = tol * np.ones(nh_e)
            ocp.constraints.lh_e = big_neg * np.ones(nh_e)
        # Slack dims
        ocp.dims.ns = len(idxsh)
        ocp.dims.ns_e = len(idxsh_e)
        ocp.constraints.idxsh = np.array(idxsh, dtype=np.int64)
        ocp.constraints.idxsh_e = np.array(idxsh_e, dtype=np.int64)
        ocp.cost.Zl = np.array(Zl)
        ocp.cost.Zu = np.array(Zu)
        ocp.cost.zl = np.array(zl)
        ocp.cost.zu = np.array(zu)
        ocp.cost.Zl_e = np.array(Zl_e)
        ocp.cost.Zu_e = np.array(Zu_e)
        ocp.cost.zl_e = np.array(zl_e)
        ocp.cost.zu_e = np.array(zu_e)
        ocp.constraints.lsh = np.zeros(len(idxsh))
        ocp.constraints.ush = np.array(max_slack_list) if max_slack_list else np.full(len(idxsh), big_pos)
        ocp.constraints.lsh_e = np.zeros(len(idxsh_e))
        ocp.constraints.ush_e = np.array(max_slack_list_e) if max_slack_list_e else np.full(len(idxsh_e), big_pos)
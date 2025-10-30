import dataclasses
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
import scipy
import torch
import yaml
from crazyflow.sim.physics import ang_vel2rpy_rates
from crazyflow.sim.symbolic import SymbolicModel
from munch import munchify
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans


def recursive_update(obj: Union[dict, object], updates: dict):
    """Recursively update a dataclass instance or dict with values from a nested dict."""
    # Convert updates to dict if it's a dataclass
    if dataclasses.is_dataclass(updates):
        updates = dataclasses.asdict(updates)
    if isinstance(obj, dict):
        for k, v in updates.items():
            if k in obj and (isinstance(obj[k], dict) or dataclasses.is_dataclass(obj[k])) and isinstance(v, dict):
                recursive_update(obj[k], v)
            else:
                obj[k] = v
    elif dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            if field.name in updates:
                value = updates[field.name]
                current = getattr(obj, field.name)
                if dataclasses.is_dataclass(current) and isinstance(value, dict):
                    recursive_update(current, value)
                elif isinstance(current, dict) and isinstance(value, dict):
                    recursive_update(current, value)
                else:
                    setattr(obj, field.name, value)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """Converts a numpy array to a torch tensor on CPU/GPU."""
    if device == "cuda":
        return torch.Tensor(arr).cuda()
    elif device == "cpu":
        return torch.Tensor(arr)
    else:
        raise RuntimeError(f"Device type {device} unknown.")


def to_numpy(t: torch.Tensor, device: str) -> np.ndarray:
    """Converts a torch tensor on CPU/GPU to a numpy array."""
    if device == "cuda":
        return t.cpu().detach().numpy()
    elif device == "cpu":
        return t.detach().numpy()
    else:
        raise RuntimeError(f"Device type {device} unknown.")


def to_torch(x, device="cpu", dtype=None):
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.to(device=device, dtype=dtype)
        else:
            x = x.to(device=device)
    else:
        if dtype is not None:
            x = torch.tensor(x, device=device, dtype=dtype)
        else:
            x = torch.tensor(x, device=device)
    return x


def get_equilibrium(
    model: SymbolicModel, eps: float = 1e-5, raise_error_if_fail: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an equilibrium (x_eq, u_eq) such that f(x_eq, u_eq) = 0.

    Uses a nonlinear least-squares formulation: minimize ||f(x,u)||^2.

    Returns:
        (x_eq, u_eq) as 1D numpy arrays.
    """
    nx, nu = model.nx, model.nu
    z = cs.MX.sym("z", nx + nu)
    x = z[:nx]
    u = z[nx:]
    f_res = model.fc_func(x, u)  # shape (nx,1)

    # Least-squares objective
    obj = cs.sumsqr(f_res)

    nlp = {"x": z, "f": obj}
    success = True
    try:
        solver = cs.nlpsol("eq", "ipopt", nlp, {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes"})
        z0 = np.zeros(nx + nu)
        sol = solver(x0=z0)
        z_sol = np.array(sol["x"]).squeeze()
        x_eq = z_sol[:nx]
        u_eq = z_sol[nx:]
        # Simple residual check; fall back to zeros if bad.
        f_val = np.array(model.fc_func(x_eq, u_eq)).squeeze()
        if np.linalg.norm(f_val) > eps:
            success = False
            x_eq = np.zeros(nx)
            u_eq = np.zeros(nu)
    except Exception as e:
        print(f"Equilibrium computation failed: {e}")
        success = False
        x_eq = np.zeros(nx)
        u_eq = np.zeros(nu)

    if not success:
        if raise_error_if_fail:
            raise RuntimeError("Equilibrium computation failed.")
        else:
            print("Warning: Equilibrium computation failed or bad residual; using zeros.")

    return x_eq, u_eq


def select_inducing_points(
    X_train,
    num_inducing_points,
    mode: str = "fps",
    kmeans_random_state: int = 0,
    seed: int = None,
    dedup_tol: float = 1e-8,
    start: str = "center",  # "center" (robust) or "rand"
):
    """Select inducing points from training data for variational inference.

    Args:
        X_train (torch.Tensor or np.ndarray): Training data of shape (N, D).
        num_inducing_points (int): Number of inducing points to select.
        mode (str): "rnd", "kmeans", "fps", or "kmeans+fps".
        kmeans_random_state (int): Random seed for k-means.
        seed (int): Random seed for reproducibility.
        dedup_tol (float): Tolerance for de-duplication on standardized features.
        start (str): FPS start strategy: "center" (closest to standardized mean) or "rand".

    Returns:
        torch.Tensor: Inducing points of shape (m, D), m = min(num_inducing_points, N).
    """
    if seed is not None:
        set_seed(seed)

    # Convert to torch tensor
    if isinstance(X_train, np.ndarray):
        X = torch.from_numpy(X_train)
    else:
        X = X_train
    X = X.detach()
    device, dtype = X.device, X.dtype
    N, D = X.shape
    m = min(num_inducing_points, N)
    if m <= 0:
        raise ValueError("num_inducing_points must be > 0 and X_train must be non-empty.")

    # Standardize for distance-based selection
    X_cpu = X.cpu().to(dtype=torch.float64)
    mean = X_cpu.mean(dim=0, keepdim=True)
    std = X_cpu.std(dim=0, unbiased=False, keepdim=True)
    std[std < 1e-12] = 1.0
    Xs = (X_cpu - mean) / std

    def fps_select(Xs_: torch.Tensor, k: int, start_idx: int | None = None, preselected: torch.Tensor | None = None):
        """Farthest-Point Sampling on standardized features; returns indices in original X."""
        N_ = Xs_.shape[0]
        selected = preselected.clone().tolist() if preselected is not None and len(preselected) > 0 else []
        if start_idx is None:
            if start == "center":
                # Start at medoid: closest to standardized mean (0)
                norms = torch.sum(Xs_**2, dim=1)
                start_idx = int(torch.argmin(norms).item())
            else:
                g = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
                start_idx = int(torch.randint(0, N_, (1,), generator=g).item())
        if start_idx not in selected:
            selected.append(start_idx)
        # Initialize distances
        dists = torch.full((N_,), float("inf"), dtype=torch.float64)
        Xs_sel = Xs_[selected[-1]].unsqueeze(0)
        dists = torch.minimum(dists, torch.cdist(Xs_, Xs_sel).squeeze(1))
        target_len = k + (len(preselected) if preselected is not None else 0)
        while len(selected) < target_len:
            idx = torch.argmax(dists).item()
            selected.append(idx)
            Xs_sel = Xs_[idx].unsqueeze(0)
            dists = torch.minimum(dists, torch.cdist(Xs_, Xs_sel).squeeze(1))
        return torch.tensor(selected[-k:], dtype=torch.long)

    def dedup_rows(Zs: torch.Tensor, Z_orig: torch.Tensor, k: int) -> torch.Tensor:
        # Round standardized data to grid, then unique
        Z_round = torch.round(Zs / dedup_tol) * dedup_tol
        uniq, idx = torch.unique(Z_round, dim=0, return_index=True)
        Z_unique = Z_orig[idx]
        if Z_unique.shape[0] >= k:
            return Z_unique[:k]
        # Fallback: farthest point sampling to fill remaining
        remaining = k - Z_unique.shape[0]
        extra = fps_select(Xs, remaining, start_idx=None, preselected=idx)
        return torch.cat([Z_unique, X[extra]], dim=0)

    mode = mode.lower()
    if mode == "rnd":
        g = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
        idx = torch.randperm(N, generator=g)[:m]
        Z = X[idx]
        Zs = Xs[idx]
        inducing_points = dedup_rows(Zs, Z, m)

    elif mode == "kmeans":
        kmeans = KMeans(n_clusters=m, random_state=kmeans_random_state, n_init=10)
        centers = torch.tensor(kmeans.fit(Xs.numpy()).cluster_centers_, dtype=torch.float64)
        # Map back to original scale
        Z_approx = centers * std + mean
        inducing_points = dedup_rows(centers, Z_approx.to(dtype=dtype, device=device), m)

    elif mode == "fps":
        idx = fps_select(Xs, m, start_idx=None)
        inducing_points = X[idx]

    elif mode == "kmeans+fps":
        # Start with kmeans centers then fill duplicates with FPS
        kmeans = KMeans(n_clusters=m, random_state=kmeans_random_state, n_init=10)
        centers = torch.tensor(kmeans.fit(Xs.numpy()).cluster_centers_, dtype=torch.float64)
        Z_approx = centers * std + mean
        Z = Z_approx.to(dtype=dtype)
        inducing_points = dedup_rows(centers, Z.to(device=device), m)

    else:
        raise ValueError("Mode must be 'rnd', 'kmeans', 'fps', or 'kmeans+fps'.")
    return inducing_points.to(device=device, dtype=dtype)


def discretize_dynamics_rk4(
    f: cs.Function, dt: float, x: cs.SX, u: cs.SX, p: cs.SX = None, as_function: bool = False
) -> Union[cs.Function, cs.SX]:
    """Discretize a continuous-time system using the Runge-Kutta 4 method.

    Args:
        f: Continuous-time dynamics function.
        dt: Time step for discretization.
        x: State variable.
        u: Control input variable.
        p: Optional parameter variable.
        as_function: If True, return a casadi Function, otherwise return the discretized state.
    """
    if p is None:
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        k1 = f(x, u, p)
        k2 = f(x + dt / 2 * k1, u, p)
        k3 = f(x + dt / 2 * k2, u, p)
        k4 = f(x + dt * k3, u, p)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if as_function:
        return cs.Function("f_disc", [x, u, p], [x_next])
    return x_next


def discretize_linear_system(
    A: np.ndarray, B: np.ndarray, dt: float, exact: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """(Non-exact) Discretization of a linear system.

    Args:
        A (np.ndarray): Continuous-time state matrix.
        B (np.ndarray): Continuous-time input matrix.
        dt (float): Time step.
        exact (bool): Whether to use exact discretization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Discrete-time state and input matrices.
    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        return Md[:state_dim, :state_dim], Md[:state_dim, state_dim:]

    return np.eye(state_dim) + A * dt, B * dt


def remove_path(path: str):
    """Remove a directory and its contents or a file if it exists."""
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def obs_to_state(obs: dict[str, npt.NDArray]) -> npt.NDArray:
    """Convert the observation from the obs dictionary to the state vector.

    Args:
        obs (dict[str, NDArray]): A dictionary containing observation data.
            Keys include:
                - "pos" (NDArray): Position vector, shape (3,).
                - "vel" (NDArray): Linear velocity vector, shape (3,).
                - "quat" (NDArray): Orientation in quaternion format, shape (4,),
                                    we use scalar-last order (x, y, z, w).
                - "rpy_rates" (NDArray): Angular velocity (roll, pitch, yaw rates), shape (3,).

    Returns:
        NDArray: A single state vector containing concatenated observation data.
    """
    # Extract position
    pos = obs["pos"].squeeze()  # shape: (3,)

    # Extract linear velocity
    vel = obs["vel"].squeeze()  # shape: (3,)

    # Extract orientation as quaternion and convert to Euler angles
    quat = obs["quat"].squeeze()  # shape: (4,)
    euler = R.from_quat(quat).as_euler("xyz")  # shape: (3,), Euler angles
    # euler = euler[::-1] # [roll, pitch, yaw]
    # Extract angular velocity
    rpy_rates = ang_vel2rpy_rates(obs["ang_vel"].squeeze(), quat)  # shape: (3,)

    # Concatenate into a single state vector
    state = np.array([pos[0], pos[1], pos[2], *euler, vel[0], vel[1], vel[2], *rpy_rates])

    return state


##################### path and config utils #####################


def setup_config_and_result_dir(config_name: str = "base_config_hidden.yaml") -> dict:
    """Load the configuration file and setup the results export folder.

    Args:
        config_name: Name of the configuration file

    Returns:
        config: the loaded configuration
    """
    config_path = Path(__file__).parent / config_name
    if not config_path.exists():
        root_dir = str(Path(__file__).parents[2])
        config_name = config_name.removesuffix(".yaml")
        config_name += "_private.yaml"
        config_path = os.path.join(root_dir, "test", "configs", config_name)

    # root of the project
    # config_path = Path(__file__).parent / "config" / config_name

    with open(config_path, "r") as file:
        config = munchify(yaml.safe_load(file))

    save_dir = Path(__file__).parents[1] / config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = save_dir
    return config


def mkdir_date(path: Path) -> Path:
    """Make a unique directory within the given directory with the current time as name.

    Args:
        path: Parent folder path

    Returns:
        the path for storing results
    """
    # assert path.is_dir(), f"Path {path} is not a directory"
    save_dir = path / datetime.now().strftime("%Y_%m_%d_%H_%M")
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        t = 1
        while save_dir.is_dir():
            curr_date_unique = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})"
            save_dir = path / (curr_date_unique)
            t += 1
        save_dir.mkdir(parents=True)
    return save_dir

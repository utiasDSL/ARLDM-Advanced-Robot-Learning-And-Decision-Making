import time
from collections import defaultdict

import crazyflow  # noqa: F401, register environments
import gymnasium
import numpy as np
import torch
from crazyflow.sim.symbolic import symbolic_attitude
from exercise05.gpmpc import GPMPC
from exercise05.plotting import make_quad_plots, plot_quad_eval
from exercise05.utils import obs_to_state, setup_config_and_result_dir
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy


def run_evaluation(env, ctrl: GPMPC, seed: int) -> dict:
    episode_data = defaultdict(list)
    ctrl.reset()
    obs, _ = env.reset(seed=seed)
    obs = obs_to_state(obs)

    episode_data["obs"].append(obs)

    env.action_space.seed(seed)
    ctrl_data = defaultdict(list)
    inference_time_data = []

    while True:
        time_start = time.perf_counter()
        action = ctrl.select_action(obs)
        inference_time_data.append(time.perf_counter() - time_start)
        # Vector environment expects a batched action (world size 1) in float32
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32).reshape(1, -1))
        obs = obs_to_state(obs)
        done = terminated or truncated
        step_data = {"obs": obs, "action": action, "done": done, "reward": reward, "length": 1}
        for key, val in step_data.items():
            episode_data[key].append(val)
        if done:
            break
    for key, val in episode_data.items():
        episode_data[key] = np.array(val)

    episode_data["controller_data"] = dict(ctrl_data)
    episode_data["inference_time_data"] = inference_time_data
    return episode_data


def extract_data(data, rng):
    """Sample data from a list of observations and actions."""
    n = data["action"].shape[0]
    n_samples = n - 1
    assert isinstance(data["action"], np.ndarray)
    assert isinstance(data["obs"], np.ndarray)
    idx = rng.choice(n - 1, n_samples, replace=False) if n_samples < n else np.arange(n - 1)
    obs = np.array(data["obs"])
    actions = np.array(data["action"])
    return obs[idx, ...], actions[idx, ...], obs[idx + 1, ...]


def learn(
    n_epochs: int,
    ctrl: GPMPC,
    env: JaxToNumpy,
    lr: float,
    gp_iterations: int,
    seed: int,
    max_samples: int,
    use_validation: bool = False,
):
    """Performs multiple epochs learning.

    Args:
        n_epochs (int): Number of epochs to run.
        ctrl (GPMPC): The controller to train.
        env (JaxToNumpy): The environment to run the controller in.
        lr (float): Learning rate for the GP training.
        gp_iterations (int): Number of iterations to train the GP.
        seed (int): Random seed for reproducibility.
        max_samples (int): Maximum number of samples to use for training.
        use_validation (bool): Whether to use validation split during GP training.

    Returns:
        train_runs (dict): Dictionary containing training runs data.
        test_runs (dict): Dictionary containing test runs data.
    """
    train_runs, test_runs = {}, {}
    # Generate n unique random integers for epoch seeds and one for evaluation
    rng = np.random.default_rng(seed)
    eval_seed = int(rng.integers(np.iinfo(np.int32).max))
    # To make the results reproducible across runs with varying number of epochs, we create seeds for 1e6 epochs and then use the first n_epochs of them. This guarantees that the same seeds  are used for the episodes no matter how many epochs are run. We could also reseed the rng after sampling and sample each episode independently, but this prevents us from using replace.
    assert n_epochs < int(1e6), f"Number of epochs must be less than 1e6, got {n_epochs}"
    epoch_seeds = rng.choice(np.iinfo(np.int32).max, size=int(1e6), replace=False)[: n_epochs + 1]

    # pbar = tqdm(range(n_epochs), desc="GP-MPC", dynamic_ncols=True)
    # Run prior
    ctrl.prior_ctrl.reset(full_reset=True)
    train_runs[0] = run_evaluation(env, ctrl.prior_ctrl, seed=int(epoch_seeds[0]))
    test_runs[0] = run_evaluation(env, ctrl.prior_ctrl, seed=eval_seed)
    # Initialize training data
    x_replay, y_replay = np.zeros((0, 7)), np.zeros((0, 3))  # 7 inputs, 3 outputs

    for epoch in range(1, n_epochs + 1):
        # Gather training data and train the GP
        ########################################################################
        # Task 3                                                               
        # TODO:                                                                
        # 1. Sample data from the previous training run using `sample_data`.   
        # 2. Preprocess the sampled data using `ctrl.preprocess_data`.         
        # 3. Obtain current training inputs and updated replay buffers via the 
        # error_based_replay_buffer() function. (Do not forget to pass the     
        # rng!)                                                                
        ########################################################################
        











        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################
        t3 = time.perf_counter()
        ctrl.train_gp(
            x=x_train,
            y=y_train,
            lr=lr,
            iterations=gp_iterations,
            val_split=0.2 if use_validation else 0.0,
        )
        t4 = time.perf_counter()
        # Test new policy.
        test_runs[epoch] = run_evaluation(env, ctrl, eval_seed)
        t5 = time.perf_counter()
        # Gather training data
        train_runs[epoch] = run_evaluation(env, ctrl, int(epoch_seeds[epoch]))
        t6 = time.perf_counter()
        # Print timing table
        print(f"\nEpoch {epoch}/{n_epochs}")
        print("\nExecution Times (seconds):")
        print(f"{'Operation':<25} {'Time (s)':<10}")
        print("-" * 35)
        print(f"{'Train GP':<25} {t4 - t3:>10.2f}")
        print(f"{'Test GPMPC Performance':<25} {t5 - t4:>10.2f}")
        print(f"{'Collect GP Data':<25} {t6 - t5:>10.2f}")
        # pbar.update(1)

    return train_runs, test_runs


def run():
    """The main function running experiments for model-based methods."""
    config = setup_config_and_result_dir()
    torch.manual_seed(config.seed)

    # Get initial state and create environments
    env = JaxToNumpy(gymnasium.make_vec("DroneFigureEightTrajectory-v0", num_envs=1))
    traj = env.unwrapped.trajectory.T
    dt = 1 / env.unwrapped.freq
    config.dt = dt
    action_space = {
        "low": env.unwrapped.single_action_space.low + 0.01,
        "high": env.unwrapped.single_action_space.high - 0.01,
    }

    # TODO: Add the information from config.gpmpc.prior_info to the symbolic model
    prior_model = symbolic_attitude(dt=dt, params=config.nominal_model_params)

    # Run the experiment.

    # Create controller.
    ctrl = GPMPC(
        prior_model,
        traj=traj,
        prior_params=config.nominal_model_params,
        output_dir=config.save_dir,
        seed=config.seed,
        action_space=action_space,
        **dict(config.gpmpc),
    )

    train_runs, test_runs = learn(ctrl=ctrl, env=env, seed=config.seed, **dict(config.learn))

    def compute_mse(ref, traj):
        max_len = min(len(ref), len(traj))
        return np.mean((ref[:max_len] - traj[:max_len]) ** 2, axis=(0, 1))

    # Compute MSE for test runs
    mse_dict = {}
    for i, run in test_runs.items():
        mse_dict[i] = compute_mse(ctrl.traj.T, run["obs"])
    for i, mse in mse_dict.items():
        print(f"MPC MSE: {mse}") if i == 0 else print(f"GP-MPC MSE {i}: {mse}")

    # plotting training and evaluation results
    make_quad_plots(
        test_runs=test_runs,
        train_runs=train_runs,
        trajectory=ctrl.traj.T,
        save_dir=config.save_dir,
        show=False,
    )

    # Run tests on a seed different from the test seed
    trajs_data = run_evaluation(env, ctrl, seed=config.seed + 1)
    env.close()

    dt = ctrl.model.dt
    plot_quad_eval(trajs_data, traj, dt, config.save_dir)


def error_based_replay_buffer(
    x_replay,
    y_replay,
    new_inputs,
    new_targets,
    rng,
    ctrl: GPMPC,
    n_samples,
    buffer_size=1000,
    alpha=0.1,
) -> tuple:
    """Update the replay buffer with new data, compute error and uncertainty scores, and select top samples for training.

    Returns:
        Tuple of (x_train, y_train, x_replay, y_replay)
    """
    # Update replay buffer with new data
    x_replay = np.vstack((x_replay, new_inputs))
    y_replay = np.vstack((y_replay, new_targets))

    # If no GP is available, return the replay buffer as is (reduce to n_samples)
    if not ctrl.gaussian_process:
        idx = rng.permutation(x_replay.shape[0])[:n_samples]
        x_replay = x_replay[idx]
        y_replay = y_replay[idx]
        return x_replay, y_replay, x_replay, y_replay

    # Truncate buffer if needed
    if x_replay.shape[0] > buffer_size:
        idx = rng.permutation(x_replay.shape[0])[:buffer_size]
        x_replay = x_replay[idx]
        y_replay = y_replay[idx]

    # Compute GP prediction error for all buffer entries
    pred_mean, pred_std = ctrl.infer_gps_only(x_replay)
    pred_mean = pred_mean.cpu().numpy()
    pred_std = pred_std.cpu().numpy()
    # Compute absolute error
    errors = np.abs(pred_mean - y_replay).sum(axis=1)
    # Compute uncertainty as the sum of standard deviations
    uncertainty = np.abs(pred_std).sum(axis=1)
    # Normalize errors and uncertainty
    norm_error = errors / (np.max(errors) + 1e-8)
    norm_uncertainty = uncertainty / (np.max(uncertainty) + 1e-8)
    # Combine error and uncertainty into a score
    score = (1 - alpha) * norm_error + alpha * norm_uncertainty

    # Select top-n_samples highest scores
    n_select = min(n_samples, x_replay.shape[0])
    idx = np.argsort(-score)[:n_select]
    x_train = x_replay[idx]
    y_train = y_replay[idx]

    return x_train, y_train, x_replay, y_replay


if __name__ == "__main__":
    tstart = time.perf_counter()
    run()
    print(f"Experiment took {time.perf_counter() - tstart:.2f} seconds")

import os
import pickle

import numpy as np
from exercise04.utils import set_seed


def collect_state_transitions(
    save_dir: str = None,
    filename: str = None,
    ppo_checkpoint_path: str = None,
    load_if_exists: bool = True,
    visualize: bool = False,
    noisy: bool = True,
    noise_factor: float = 1.0,
):
    """Collects state transitions from the DroneFigureEightTrajectory-v0 environment using a pretrained PPO policy and saves them to a file.

    Args:
        save_dir (str): Directory where the data will be saved.
        filename (str): Name of the file to save the data.
        ppo_checkpoint_path (str): Path to the policy checkpoint that should be used for data collection.
        load_if_exists (bool): If True, load existing data from disk if available.
        visualize (bool): If True, render the environment during data collection.
        noisy (bool): If True, adds noise to actions and observations.
        noise_factor (float): Factor to scale the noise added to actions and observations.

    Returns:
        dict: Dictionary containing states, actions, and next_states.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "outputs")
    if filename is None:
        filename = "state_transitions.pkl"

    data_path = os.path.join(save_dir, filename)
    data_exists = os.path.exists(data_path)
    if load_if_exists and data_exists:
        with open(data_path, "rb") as f:
            recorded_data = pickle.load(f)
        print("Loaded existing data from disk.")
        return recorded_data

    # Extra imports
    import gymnasium
    import gymnasium.wrappers.vector.jax_to_torch
    import jax
    import torch
    from crazyflow.envs.norm_actions_wrapper import NormalizeActions
    from exercise06.agent import Agent
    from exercise06.wrappers import FlattenJaxObservation

    if ppo_checkpoint_path is None:
        this_dir = os.path.dirname(__file__)
        ppo_checkpoint_path = os.path.join(this_dir, "ppo_checkpoint.pt")

    assert os.path.exists(ppo_checkpoint_path), (
        f"Checkpoint file {ppo_checkpoint_path} does not exist. "
        "Please provide a valid path to the PPO checkpoint."
    )

    print("Collect data...")
    recorded_data = {"states": [], "actions": [], "next_states": []}

    # Parameters for data collection
    n_episodes = 12

    # Configuration
    device = "cpu"  # For few numbers of environments cpu is faster
    # print(f"Using device: {device}")
    n_envs = 2

    # Seeding
    seed = 42
    set_seed(seed)

    # Create and wrap environment
    env = gymnasium.make_vec(
        "DroneFigureEightTrajectory-v0",
        freq=50,
        num_envs=n_envs,
        device=device,
    )
    env = NormalizeActions(env)
    env = FlattenJaxObservation(env)
    norm_env = gymnasium.wrappers.vector.NormalizeObservation(env)
    norm_env.update_running_mean = False
    env = gymnasium.wrappers.vector.jax_to_torch.JaxToTorch(norm_env, device=device)

    # Load policy
    checkpoint = torch.load(
        ppo_checkpoint_path, weights_only=True, map_location=torch.device(device)
    )

    # Create agent and load state
    agent = Agent(env).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])

    # Set normalization parameters
    jax_device = jax.devices(device)[0]
    norm_env.obs_rms.mean = jax.dlpack.from_dlpack(checkpoint["obs_mean"], jax_device)
    norm_env.obs_rms.var = jax.dlpack.from_dlpack(checkpoint["obs_var"], jax_device)

    # collect state transitions; this can take a minute
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = torch.zeros(n_envs, dtype=bool, device=device)

        while not done.all():
            with torch.no_grad():
                action, _, _, _ = agent.action_and_value(obs, deterministic=True)
            recorded_data["states"].append(
                obs[0, :13].cpu().numpy()
            )  # we know that first 13 entries are the states of the drone
            obs, _, terminated, truncated, _ = env.step(action)
            recorded_data["actions"].append(action[0].cpu().numpy())
            recorded_data["next_states"].append(
                obs[0, :13].cpu().numpy()
            )  # we know that first 13 entries are the states of the drone
            if episode == 0 and visualize:
                env.render()
            done = terminated | truncated

    # transform lists to numpy arrays
    for key in recorded_data.keys():
        recorded_data[key] = np.array(recorded_data[key])

    if noisy:
        # Add noise to actions and observations
        action_noise_std = 0.01 * noise_factor
        state_noise_std = 0.01 * noise_factor
        recorded_data["actions"] += np.random.normal(
            0, action_noise_std, recorded_data["actions"].shape
        )
        recorded_data["states"] += np.random.normal(
            0, state_noise_std, recorded_data["states"].shape
        )
        recorded_data["next_states"] += np.random.normal(
            0, state_noise_std, recorded_data["next_states"].shape
        )

    # print("Shape of states:", recorded_data["states"].shape)
    # print("Shape of actions:", recorded_data["actions"].shape)

    # Save data
    os.makedirs(save_dir, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(recorded_data, f)
    print(f"Saved data to disk in {save_dir}.")
    return recorded_data

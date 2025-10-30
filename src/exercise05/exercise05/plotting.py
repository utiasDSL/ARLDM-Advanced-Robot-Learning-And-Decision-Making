"""GP-MPC lotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

state_names = ["x", "y", "z", "roll", "pitch", "yaw", "dx", "dy", "dz", "droll", "dpitch", "dyaw"]


def get_runtime(test_runs, train_runs):
    """Get the mean, std, and max runtime."""
    # NOTE: only implemented for single episode
    # NOTE: the first step is popped out because of the ipopt initial guess

    num_epochs = len(train_runs.keys())
    num_train_samples_by_epoch = []  # number of training data
    mean_runtime = np.zeros(num_epochs)
    std_runtime = np.zeros(num_epochs)
    max_runtime = np.zeros(num_epochs)

    runtime = []
    for epoch in range(num_epochs):
        num_samples = len(train_runs[epoch].keys())
        num_train_samples_by_epoch.append(num_samples)
        runtime = test_runs[epoch]["inference_time_data"][1:]  # remove the first step

        mean_runtime[epoch] = np.mean(runtime)
        std_runtime[epoch] = np.std(runtime)
        max_runtime[epoch] = np.max(runtime)

    runtime_result = {
        "mean": mean_runtime,
        "std": std_runtime,
        "max": max_runtime,
        "num_train_samples": num_train_samples_by_epoch,
    }
    return runtime_result


def plot_runtime(runtime, num_points_per_epoch, save_dir: Path):
    mean_runtime = runtime["mean"]
    std_runtime = runtime["std"]
    max_runtime = runtime["max"]
    # num_train_samples = runtime['num_train_samples']
    plt.plot(num_points_per_epoch, mean_runtime, label="mean")
    plt.fill_between(
        num_points_per_epoch, mean_runtime - std_runtime, mean_runtime + std_runtime, alpha=0.3, label="1-std"
    )
    plt.plot(num_points_per_epoch, max_runtime, label="max", color="r")
    plt.legend()
    plt.xlabel("Train Steps")
    plt.ylabel("Runtime (s) ")
    plt.title("GP-MPC Runtime")
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir / "runtime.pdf")
        plt.cla()
        plt.clf()
        data = np.vstack((num_points_per_epoch, mean_runtime, std_runtime, max_runtime)).T
        np.savetxt(save_dir / "runtime.csv", data, delimiter=",", header="Train Steps, Mean, Std, Max")


def plot_runs(all_runs, num_epochs, ind=0, ylabel="x position", save_dir: Path | None = None, traj=None):
    # plot the reference trajectory
    if traj is not None:
        plt.plot(traj[:, ind], label="Reference", color="gray", linestyle="--")
    # plot the prior controller
    plt.plot(all_runs[0]["obs"][:, ind], label="prior MPC")
    # plot each learning epoch
    for epoch in range(1, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch]["obs"][:, ind], label=f"GP-MPC {epoch}")
    plt.title(ylabel)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir / f"x{ind}.pdf")
    else:
        plt.show()
    plt.cla()
    plt.clf()


def plot_runs_input(all_runs, num_epochs, ind=0, ylabel="x position", save_dir: Path | None = None):
    # plot the prior controller
    plt.plot(all_runs[0]["action"][:, ind], label="prior MPC")
    # plot each learning epoch
    for epoch in range(1, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch]["action"][:, ind], label=f"GP-MPC {epoch}")
    plt.title(ylabel)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir / f"u{ind}.pdf")
    else:
        plt.show()
    plt.clf()


def plot_learning_curve(avg_rewards, num_points_per_epoch, stem, save_dir: Path):
    samples = num_points_per_epoch  # data number
    rewards = np.array(avg_rewards)
    plt.plot(samples, rewards)
    plt.title("Avg Episode" + stem)
    plt.xlabel("Training Steps")
    plt.ylabel(stem)
    plt.savefig(save_dir / (stem + ".pdf"))
    plt.cla()
    plt.clf()
    data = np.vstack((samples, rewards)).T
    np.savetxt(save_dir / (stem + ".csv"), data, delimiter=",", header="Train steps,Cost")


def plot_trajectory(runs, traj, first_label="Prior MPC", second_label="GP-MPC"):
    num_steps = runs[0]["obs"].shape[0]
    # trim the traj steps to mach the evaluation steps
    traj = traj[0:num_steps, :]
    num_epochs = len(runs)

    _ = plt.figure(figsize=(10, 6))

    # x-z plane
    idx = [0, 2]
    plt.plot(traj[idx[0], :], traj[idx[1], :], label="Reference", color="gray", linestyle="-")
    plt.plot(runs[0]["obs"][:, idx[0]], runs[0]["obs"][:, idx[1]], label=f"{first_label}")
    for epoch in range(1, num_epochs):
        plt.plot(runs[epoch]["obs"][:, idx[0]], runs[epoch]["obs"][:, idx[1]], label=f"{second_label} epoch %s" % epoch)
    plt.title("X-Z plane path")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.legend()


def plot_xyz_trajectory(runs, ref, save_dir: Path):
    """Plot the x-y, x-z, and y-z trajectories.

    Args:
        runs (list): List of dictionaries containing the observations.
        ref (ndarray): Reference trajectory.
        save_dir (Path): Directory to save the plots.
        show (bool): Whether to show the plots.
    """
    num_epochs = len(runs)
    fig, ax = plt.subplots(3, 1)

    # x-y plane
    ax[0].plot(ref[:, 0], ref[:, 2], label="Reference", color="gray", linestyle="--")
    ax[0].plot(runs[0]["obs"][:, 0], runs[0]["obs"][:, 2], label="prior MPC")
    for epoch in range(1, num_epochs):
        ax[0].plot(runs[epoch]["obs"][:, 0], runs[epoch]["obs"][:, 2], label="GP-MPC %s" % epoch)
    ax[0].set_title("X-Y plane path")
    ax[0].set_xlabel("X [m]")
    ax[0].set_ylabel("Y [m]")
    ax[0].legend()
    # x-z plane
    ax[1].plot(ref[:, 0], ref[:, 4], label="Reference", color="gray", linestyle="--")
    ax[1].plot(runs[0]["obs"][:, 0], runs[0]["obs"][:, 4], label="prior MPC")
    for epoch in range(1, num_epochs):
        ax[1].plot(runs[epoch]["obs"][:, 0], runs[epoch]["obs"][:, 4], label="GP-MPC %s" % epoch)
    ax[1].set_title("X-Z plane path")
    ax[1].set_xlabel("X [m]")
    ax[1].set_ylabel("Z [m]")
    ax[1].legend()
    # y-z plane
    ax[2].plot(ref[:, 2], ref[:, 4], label="Reference", color="gray", linestyle="--")
    ax[2].plot(runs[0]["obs"][:, 2], runs[0]["obs"][:, 4], label="prior MPC")
    for epoch in range(1, num_epochs):
        ax[2].plot(runs[epoch]["obs"][:, 2], runs[epoch]["obs"][:, 4], label="GP-MPC %s" % epoch)
    ax[2].set_title("Y-Z plane path")
    ax[2].set_xlabel("Y [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].legend()

    fig.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        plt.close(fig)
        fig.savefig(save_dir / "xyz_path.pdf")
        plt.cla()
        plt.clf()


def make_quad_plots(test_runs, train_runs, trajectory, save_dir, show=True):
    if show:
        save_dir = None
        fig_dir = None
    else:
        fig_dir = save_dir / "figs"
        fig_dir.mkdir(parents=True, exist_ok=True)

    num_steps, nx = test_runs[0]["obs"].shape
    nu = test_runs[0]["action"].shape[1]
    # trim the traj steps to mach the evaluation steps
    trajectory = trajectory[0:num_steps, :]
    num_epochs = len(test_runs)

    num_points_per_epoch = []
    plot_xyz_trajectory(test_runs, trajectory, fig_dir)
    for ind in range(nx):
        plot_runs(test_runs, num_epochs, ind=ind, ylabel=f"x{ind}", save_dir=fig_dir, traj=trajectory)
    for ind in range(nu):
        plot_runs_input(test_runs, num_epochs, ind=ind, ylabel=f"u{ind}", save_dir=fig_dir)
    num_points = 0
    num_points_per_epoch.append(num_points)
    for epoch in range(1, num_epochs):
        num_points += train_runs[epoch]["obs"].shape[0]
        num_points_per_epoch.append(num_points)

    runtime_result = get_runtime(test_runs, train_runs)
    plot_runtime(runtime_result, num_points_per_epoch, fig_dir)


def plot_quad_eval(trajectories, reference, dt: float, save_path: Path):
    """Plots the input and states to determine success."""
    state_stack = trajectories["obs"]
    input_stack = trajectories["action"]
    nx = state_stack.shape[1]

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, dt * plot_length, plot_length)

    state_labels = ["x", "d_x", "y", "d_y", "z", "d_z", "phi", "theta", "psi", "d_phi", "d_theta", "d_psi"]
    assert len(state_labels) == nx

    # Plot states
    fig, axs = plt.subplots(nx, figsize=(8, nx * 1))
    for k in range(nx):
        axs[k].plot(times, state_stack.T[k, 0:plot_length], label="actual")
        axs[k].plot(times, reference[k, 0:plot_length], color="r", label="desired")
        axs[k].set(ylabel=state_labels[k])
        axs[k].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if k != nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title("State Trajectories")
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc="lower right")
    axs[-1].set(xlabel="time (sec)")
    fig.tight_layout()

    plt.savefig(save_path / "state_trajectories.pdf")

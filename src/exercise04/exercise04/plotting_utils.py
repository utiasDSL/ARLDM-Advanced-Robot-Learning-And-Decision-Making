import math

import matplotlib.pyplot as plt
import numpy as np

_state_names = [
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
]
_action_names = ["thrust", "roll", "pitch", "yaw"]

def plot_grid(plot_fcns, titles=None, figsize=(12, 5), grid: tuple =None):
    """Plot multiple plots in an grid (n, 2) by default where n is computed.

    Args:
        plot_fcns: List of functions, each accepting an axis (ax) to plot on.
        titles: Optional list of subplot titles.
        figsize: Figure size.
        grid: Optional (n, m) tuple specifying grid dimensions.
    """
    if grid is not None:
        n, m = grid
        assert n * m >= len(plot_fcns), "Grid size too small for number of plots."
    else:
        m = 2
        n = math.ceil(len(plot_fcns) / m)
        m = m if n > 1 else 1  # If only one row, use one column

    fig, axes = plt.subplots(n, m, figsize=figsize)
    axes = axes.flatten()
    for i, plot_fcn in enumerate(plot_fcns):
        plot_fcn(axes[i])
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    # Hide unused axes if odd number of plots
    for j in range(len(plot_fcns), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_recorded_states(data, state_names=_state_names, title="Drone Positions", freq=50, figsize=(10,6), axis=None):
    """Plot recorded states."""
    if "states" in data:
        data = data["states"]
    indices = [i for i, name in enumerate(_state_names) if name in state_names]
    assert len(indices) > 0, "No valid state names provided for plotting."
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis

    if len(indices) == 2:
        ax.scatter(
            data[:, indices[0]],
            data[:, indices[1]],
            color="blue",
            alpha=0.5,
            s=2,
            label=f"{_state_names[indices[1]]} vs {_state_names[indices[0]]}",
        )
        ax.plot(data[:, indices[0]], data[:, indices[1]], color="black", alpha=0.3)
        ax.set_xlabel(f"{_state_names[indices[0]]}")
        ax.set_ylabel(f"{_state_names[indices[1]]}")
    else:
        timesteps = np.arange(len(data)) / freq  # Convert to seconds
        for idx in indices:
            ax.plot(timesteps, data[:, idx], label=_state_names[idx])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value")
        ax.legend()

    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if axis is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plot_regression_results(X, y, model_predictions, model_std=None, title="Regression Results", figsize=(8,5), axis=None):
    """Visualize regression results and uncertainty."""
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis
    ax.scatter(X, y, color="blue", alpha=0.5, label="Data")
    ax.plot(X, model_predictions, color="red", label="Prediction")
    if model_std is not None:
        ax.fill_between(
            np.ravel(X),
            model_predictions - 2 * model_std,
            model_predictions + 2 * model_std,
            color="red",
            alpha=0.2,
            label="95% CI",
        )
    ax.set_title(title)
    ax.set_xlabel("Input")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(True)
    if axis is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plot_learning_curves(train_losses, val_losses=None, figsize=(8,5), axis=None):
    """Plot training (and optionally validation) loss curves."""
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis
    ax.plot(train_losses, label="Train Loss")
    if val_losses is not None:
        ax.plot(val_losses, label="Validation Loss")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    if axis is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plot_random_test_point(y_test, y_pred, y_std, state_1_name="pos_x", state_2_name="pos_y", print_results=False, figsize=(7,6), axis=None):
    """Plot a randomly selected test point with prediction and uncertainty."""
    idx = np.random.randint(len(y_test))
    assert state_1_name in _state_names, f"State name {state_1_name} not recognized."
    assert state_2_name in _state_names, f"State name {state_2_name} not recognized."
    state_1_idx = _state_names.index(state_1_name)
    state_2_idx = _state_names.index(state_2_name)
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis
    ax.scatter(
        y_test[idx, state_1_idx], y_test[idx, state_2_idx], s=80, color="green", edgecolor="black", label="Ground Truth"
    )
    ax.scatter(
        y_pred[idx, state_1_idx], y_pred[idx, state_2_idx], s=80, color="red", edgecolor="black", label="Prediction"
    )
    ax.errorbar(
        y_pred[idx, state_1_idx],
        y_pred[idx, state_2_idx],
        xerr=y_std[idx, state_1_idx],
        yerr=y_std[idx, state_2_idx],
        fmt="o",
        color="black",
        alpha=0.7,
        capsize=4,
        label="Uncertainty",
    )
    ax.set_title(f"Test Point #{idx}: Prediction vs Ground Truth")
    ax.set_xlabel(state_1_name)
    ax.set_ylabel(state_2_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if print_results:
        print(f"Index: {idx}")
        print(f"Ground truth: ({y_test[idx, state_1_idx]:.2f}, {y_test[idx, state_2_idx]:.2f})")
        print(f"Prediction:  ({y_pred[idx, state_1_idx]:.2f}, {y_pred[idx, state_2_idx]:.2f})")
        print(f"Std:        ({y_std[idx, state_1_idx]:.2f}, {y_std[idx, state_2_idx]:.2f})")
    if axis is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plot_2d_positions_with_std(
    y_train,
    y_test,
    y_pred,
    y_std,
    state_1_name="pos_x",
    state_2_name="pos_y",
    show_train=True,
    x_lim=None,
    y_lim=None,
    plot_std=True,
    figsize=(10,8),
    axis=None,
):
    """Plot 2D positions with optional uncertainty, using state names."""
    state_1_idx = _state_names.index(state_1_name)
    state_2_idx = _state_names.index(state_2_name)
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis

    pred_mask = np.ones(len(y_pred), dtype=bool)
    train_mask = np.ones(len(y_train), dtype=bool) if y_train is not None else None
    test_mask = np.ones(len(y_test), dtype=bool) if y_test is not None else None

    if x_lim is not None:
        pred_mask &= (y_pred[:, state_1_idx] >= x_lim[0]) & (y_pred[:, state_1_idx] <= x_lim[1])
        if y_train is not None:
            train_mask &= (y_train[:, state_1_idx] >= x_lim[0]) & (y_train[:, state_1_idx] <= x_lim[1])
        if y_test is not None:
            test_mask &= (y_test[:, state_1_idx] >= x_lim[0]) & (y_test[:, state_1_idx] <= x_lim[1])
    if y_lim is not None:
        pred_mask &= (y_pred[:, state_2_idx] >= y_lim[0]) & (y_pred[:, state_2_idx] <= y_lim[1])
        if y_train is not None:
            train_mask &= (y_train[:, state_2_idx] >= y_lim[0]) & (y_train[:, state_2_idx] <= y_lim[1])
        if y_test is not None:
            test_mask &= (y_test[:, state_2_idx] >= y_lim[0]) & (y_test[:, state_2_idx] <= y_lim[1])

    if show_train and y_train is not None:
        ax.scatter(
            y_train[train_mask, state_1_idx],
            y_train[train_mask, state_2_idx],
            s=10,
            color="blue",
            alpha=0.5,
            label="Training Data",
        )
    if y_test is not None:
        ax.scatter(
            y_test[test_mask, state_1_idx],
            y_test[test_mask, state_2_idx],
            s=15,
            color="green",
            alpha=0.7,
            label="Ground Truth",
        )
    if plot_std and y_std is not None:
        ax.errorbar(
            y_pred[pred_mask, state_1_idx],
            y_pred[pred_mask, state_2_idx],
            xerr=y_std[pred_mask, state_1_idx],
            yerr=y_std[pred_mask, state_2_idx],
            fmt="o",
            color="black",
            alpha=0.1,
            label="Prediction Uncertainty",
        )
    ax.scatter(
        y_pred[pred_mask, state_1_idx],
        y_pred[pred_mask, state_2_idx],
        s=20,
        color="red",
        alpha=0.9,
        label="Predictions",
    )
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_title(f"2D Position Plot: {state_1_name} vs {state_2_name}")
    ax.set_xlabel(state_1_name)
    ax.set_ylabel(state_2_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    if axis is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plot_prediction_vs_truth(y_test, y_pred, y_std, feature_name=_state_names[0], model_name="GP", sort=True, figsize=(10,6), axis=None):
    """Plot prediction vs ground truth for a feature, with uncertainty."""
    feature_names = _state_names + _action_names
    assert feature_name in feature_names, f"Feature name {feature_name} not recognized."
    feature_idx = feature_names.index(feature_name)
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis
    idx = np.argsort(y_test[:, feature_idx]) if sort else np.arange(len(y_test))
    x = np.arange(len(idx))
    y_true = y_test[idx, feature_idx]
    y_predicted = y_pred[idx, feature_idx]
    uncertainty = y_std[idx, feature_idx]
    ax.plot(x, y_true, "b-", label="Ground Truth")
    ax.plot(x, y_predicted, "r-", label=f"{model_name} Prediction")
    ax.fill_between(
        x, y_predicted - 2 * uncertainty, y_predicted + 2 * uncertainty, color="r", alpha=0.2, label="95% CI"
    )
    ax.set_title(f"{model_name}: Prediction vs Truth ({feature_name})")
    ax.set_xlabel("Test Sample")
    ax.set_ylabel(feature_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if axis is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plot_error_distribution(y_test, y_pred, state_names=_state_names, model_name="GP", figsize=(10,5), axis=None):
    """Plot mean absolute error for each state dimension."""
    mean_errors = np.mean(np.abs(y_test - y_pred), axis=0)
    if axis is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axis
    _ = ax.bar(state_names, mean_errors, color=plt.cm.viridis(mean_errors / mean_errors.max()))
    ax.set_title(f"{model_name} Mean Absolute Error by State")
    ax.set_xlabel("State Dimension")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=45, ha="right")
    plt.tight_layout()
    if axis is None:
        plt.show()
    else:
        return ax

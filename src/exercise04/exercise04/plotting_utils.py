import matplotlib.pyplot as plt
import numpy as np


def plot_random_test_point(
    y_test, y_pred, y_std, state_names, state_1_idx=0, state_2_idx=1, print_results=True
):
    # Randomly select one point
    random_idx = np.random.randint(0, len(y_test))

    # Create a plot
    plt.figure(figsize=(10, 8))

    # Plot the selected ground truth point
    plt.scatter(
        y_test[random_idx, state_1_idx],
        y_test[random_idx, state_2_idx],
        s=100,
        color="green",
        edgecolors="black",
        label="Ground Truth",
    )

    # Plot the selected predicted point
    plt.scatter(
        y_pred[random_idx, state_1_idx],
        y_pred[random_idx, state_2_idx],
        s=100,
        color="red",
        edgecolors="black",
        label="Prediction",
    )

    # Plot the standard deviation as error bars for the selected point
    plt.errorbar(
        y_pred[random_idx, state_1_idx],
        y_pred[random_idx, state_2_idx],
        xerr=y_std[random_idx, state_1_idx],
        yerr=y_std[random_idx, state_2_idx],
        fmt="o",
        color="black",
        alpha=0.7,
        capsize=5,
        label="Prediction Uncertainty",
    )

    # Add point index as title
    plt.title(f"Data Point #{random_idx} with Prediction and Ground Truth")
    plt.xlabel(state_names[state_1_idx])
    plt.ylabel(state_names[state_2_idx])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Print only the specific position indices
    if print_results:
        print(f"Selected point index: {random_idx}")
        print(
            f"Ground truth (x, z): ({y_test[random_idx, state_1_idx]:.2f}, {y_test[random_idx, state_2_idx]:.2f})"
        )
        print(
            f"Prediction (x, z): ({y_pred[random_idx, state_1_idx]:.2f}, {y_pred[random_idx, state_2_idx]:.2f})"
        )
        print(
            f"Standard deviation (x, z): ({y_std[random_idx, state_1_idx]:.2f}, {y_std[random_idx, state_2_idx]:.2f})"
        )
    return plt


def plot_2d_positions_with_std(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    y_std,
    state_names,
    state_1_idx=0,
    state_2_idx=1,
    show_train=True,
    x_lim=None,
    y_lim=None,
    plot_std=True,
):
    if y_std is None:
        plot_std = False
    plt.figure(figsize=(10, 8))

    # Create masks for filtering data
    pred_mask = np.ones(len(y_pred), dtype=bool)
    train_mask = np.ones(len(y_train), dtype=bool) if y_train is not None else None
    test_mask = np.ones(len(y_test), dtype=bool) if y_test is not None else None

    # Apply x-axis limits if provided
    if x_lim is not None:
        pred_mask = (
            pred_mask & (y_pred[:, state_1_idx] >= x_lim[0]) & (y_pred[:, state_1_idx] <= x_lim[1])
        )
        if y_train is not None:
            train_mask = (
                train_mask
                & (y_train[:, state_1_idx] >= x_lim[0])
                & (y_train[:, state_1_idx] <= x_lim[1])
            )
        if y_test is not None:
            test_mask = (
                test_mask
                & (y_test[:, state_1_idx] >= x_lim[0])
                & (y_test[:, state_1_idx] <= x_lim[1])
            )

    # Apply y-axis limits if provided
    if y_lim is not None:
        pred_mask = (
            pred_mask & (y_pred[:, state_2_idx] >= y_lim[0]) & (y_pred[:, state_2_idx] <= y_lim[1])
        )
        if y_train is not None:
            train_mask = (
                train_mask
                & (y_train[:, state_2_idx] >= y_lim[0])
                & (y_train[:, state_2_idx] <= y_lim[1])
            )
        if y_test is not None:
            test_mask = (
                test_mask
                & (y_test[:, state_2_idx] >= y_lim[0])
                & (y_test[:, state_2_idx] <= y_lim[1])
            )

    # Plot training data if requested
    if show_train and y_train is not None:
        plt.scatter(
            y_train[train_mask, state_1_idx],
            y_train[train_mask, state_2_idx],
            s=10,
            color="blue",
            alpha=0.5,
            label="Training Data",
        )

    # Plot ground truth test data
    if y_test is not None:
        plt.scatter(
            y_test[test_mask, state_1_idx],
            y_test[test_mask, state_2_idx],
            s=15,
            color="green",
            alpha=0.7,
            label="Ground Truth",
        )

    # Plot the standard deviation as error bars (only for filtered points)
    title = "2D Position Plot"
    if plot_std:
        title = "2D Position Plot with Std"
        plt.errorbar(
            y_pred[pred_mask, state_1_idx],
            y_pred[pred_mask, state_2_idx],
            xerr=y_std[pred_mask, state_1_idx],
            yerr=y_std[pred_mask, state_2_idx],
            fmt="o",
            color="black",
            alpha=0.1,
            label="Prediction Uncertainty",
        )

    # Plot test predictions (only filtered points)
    plt.scatter(
        y_pred[pred_mask, state_1_idx],
        y_pred[pred_mask, state_2_idx],
        s=20,
        color="red",
        alpha=0.9,
        label="Predictions",
    )

    # Set plot bounds to match filter limits (if provided)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.title(title)
    plt.xlabel(state_names[state_1_idx])
    plt.ylabel(state_names[state_2_idx])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    return plt


def plot_prediction_vs_truth(y_test, y_pred, y_std, feature_idx, feature_name, model_name="GP"):
    plt.figure(figsize=(10, 6))

    # Sort by ground truth for clearer visualization
    sort_idx = np.argsort(y_test[:, feature_idx])

    x = np.arange(len(sort_idx))
    y_true = y_test[sort_idx, feature_idx]
    y_predicted = y_pred[sort_idx, feature_idx]
    uncertainty = y_std[sort_idx, feature_idx]

    plt.plot(x, y_true, "b-", label="Ground Truth")
    plt.plot(x, y_predicted, "r-", label=model_name + " Prediction")
    plt.fill_between(
        x,
        y_predicted - 2 * uncertainty,
        y_predicted + 2 * uncertainty,
        color="r",
        alpha=0.2,
        label="95% Confidence",
    )

    plt.title(f"{model_name} Prediction vs Ground Truth for {feature_name}")
    plt.xlabel("Test Sample (sorted by increasing x)")
    plt.ylabel(feature_name)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt


def plot_error_distribution(y_test, y_pred, state_names, model_name="GP"):
    errors = np.abs(y_test - y_pred)
    mean_errors = np.mean(errors, axis=0)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(state_names, mean_errors)

    # Color bars by error magnitude
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(mean_errors[i] / max(mean_errors)))

    plt.title(f"{model_name} Mean Absolute Error by State Dimension")
    plt.xlabel("State Dimension")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return plt

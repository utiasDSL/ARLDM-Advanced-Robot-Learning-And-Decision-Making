import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Normalizer:
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X_train, y_train):
        """Calculate mean and std from the training data (both X and y)."""
        if not isinstance(X_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
            raise TypeError("X_train and y_train should be PyTorch tensors.")

        # Normalize X
        self.X_mean = X_train.mean(dim=0)
        self.X_std = X_train.std(dim=0)

        # Normalize y
        self.y_mean = y_train.mean()
        self.y_std = y_train.std()

    def transform(self, X, y=None):
        """Normalize X and optionally y."""
        if not isinstance(X, torch.Tensor):
            raise TypeError("X should be a PyTorch tensor.")

        X_normalized = (X - self.X_mean) / self.X_std

        if y is not None:
            if not isinstance(y, torch.Tensor):
                raise TypeError("y should be a PyTorch tensor.")
            y_normalized = (y - self.y_mean) / self.y_std
            return X_normalized, y_normalized

        return X_normalized

    def inverse_transform(self, X, y=None):
        """Un-normalize X and optionally y."""
        if not isinstance(X, torch.Tensor):
            raise TypeError("X should be a PyTorch tensor.")

        X_unnormalized = X * self.X_std + self.X_mean

        if y is not None:
            if not isinstance(y, torch.Tensor):
                raise TypeError("y should be a PyTorch tensor.")
            y_unnormalized = y * self.y_std + self.y_mean
            return X_unnormalized, y_unnormalized

        return X_unnormalized


class CustomDataset:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        ########################################################################
        # Task 2                                                               
        # TODO:                                                                
        # 1. If shuffle is True, randomly permute self.X and self.y            
        # 2. Create batches of batch_size = self.batch_size                    
        # Hints:                                                               
        # 1. Use the yield keyword to return batches of X and y                
        # 2. If you have troubles, check the docs for iterable datasets:       
        # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset                                                        
        ########################################################################
        











        pass
        ########################################################################
        #                           END OF YOUR CODE                           
        ########################################################################


def generate_synthetic_data(n_samples=100, noise_level=0.1, seed=0):
    """Generate 1D data with mixed patterns."""
    X = np.linspace(-1, 1, n_samples).reshape(-1, 1)

    # Combine periodic and local patterns
    y = (
        np.sin(6 * X)  # High-frequency component
        + np.exp(-10 * (X - 0.5) ** 2)  # Local bump
        + 0.1 * np.random.randn(n_samples, 1)
    )  # Noise

    return X, y


def plot_regression_results(X, y, model_predictions, model_std=None, title="Regression Results"):
    """Plot the regression results with optional uncertainty."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
    plt.plot(X, model_predictions, color="red", label="Model prediction")

    if model_std is not None:
        plt.fill_between(
            X.flatten(),
            model_predictions - 2 * model_std,
            model_predictions + 2 * model_std,
            color="red",
            alpha=0.2,
            label="95% confidence interval",
        )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_curves(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_test_split(X, y, test_size):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def error_statistics(y_true, y_pred, y_std, model_name="GP") -> tuple:
    """Compute and print error statistics for regression results.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.
        y_std (np.ndarray): Predicted standard deviations (uncertainty).
        model_name (str): Name of the model (default: "GP").

    Returns:
        tuple: (overall_mae, overall_rmse, overall_uncertainty)
    """
    abs_error = np.abs(y_true - y_pred)
    squared_error = np.square(y_true - y_pred)
    # Calculate mean error for each dimension
    mean_abs_error = np.mean(abs_error, axis=0)
    rmse = np.sqrt(np.mean(squared_error, axis=0))

    # Calculate mean uncertainty (standard deviation)
    mean_uncertainty = np.mean(y_std, axis=0)
    overall_uncertainty = np.mean(mean_uncertainty)

    # Calculate overall metrics
    overall_mae = np.mean(mean_abs_error)
    overall_rmse = np.mean(rmse)

    # Print results
    print(f"{model_name} results:")
    print(f"Mean Absolute Error: {overall_mae:.4f}")
    print(f"Root Mean Square Error: {overall_rmse:.4f}")
    return overall_mae, overall_rmse, overall_uncertainty

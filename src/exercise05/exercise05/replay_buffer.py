import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class ReplayBufferConfig:
    max_size: int = 1000
    data_selection: str = "kmeans"  # Options: 'fifo', 'rnd', 'kmeans'


class ReplayBuffer:
    """Maintains a bounded dataset for online adaptation.

    Args:
        max_size: Maximum number of data points to store.
        data_selection: Strategy for data selection.
            "fifo"   -> keep most recent samples (sliding window).
            "rnd"    -> keep a random subset of size max_size.
            "kmeans" -> keep points closest to k-means cluster centers over states x.
    """

    def __init__(self, cfg: Union[ReplayBufferConfig, Dict] = {}, verbosity: str = "info"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, verbosity.upper(), logging.INFO))
        cfg = cfg if isinstance(cfg, ReplayBufferConfig) else ReplayBufferConfig(**cfg)
        assert cfg.data_selection in {"fifo", "rnd", "kmeans"}, "data_selection must be one of {'fifo','rnd','kmeans'}"
        self.max_size = cfg.max_size
        self.data_selection = cfg.data_selection
        self.data_x = deque()
        self.data_y = deque()

    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Add a batch of samples. Applies selection policy if capacity exceeded."""
        for xi, yi in zip(x, y):
            # FIFO handles overflow immediately by dropping oldest
            if self.data_selection == "fifo" and len(self.data_x) >= self.max_size:
                self.data_x.popleft()
                self.data_y.popleft()

            self.data_x.append(xi.detach().clone())
            self.data_y.append(yi.detach().clone())

            # For non-FIFO strategies, prune only after exceeding capacity
            if self.data_selection != "fifo" and len(self.data_x) > self.max_size:
                self.select_data()

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.data_x) == 0:
            return (torch.empty(0), torch.empty(0))
        return (torch.stack(list(self.data_x)), torch.stack(list(self.data_y)))

    def select_data(self) -> None:
        """Apply data selection strategy when buffer exceeds capacity."""
        n = len(self.data_x)
        if n <= self.max_size:
            return

        if self.data_selection == "fifo":
            # Already enforced in add()
            while len(self.data_x) > self.max_size:
                self.data_x.popleft()
                self.data_y.popleft()
            return

        # Convert deques to indexed lists for selection
        xs = list(self.data_x)
        ys = list(self.data_y)

        if self.data_selection == "rnd":
            # Random subset of size max_size
            # Use torch for reproducibility if user set seeds
            perm = torch.randperm(n)[: self.max_size].tolist()
            keep_indices = perm

        elif self.data_selection == "kmeans":
            # Cluster over x (flattened) and pick representative (closest) sample per cluster
            X_tensor = torch.stack(xs)  # (N, *Dx)
            X_feat = X_tensor.view(n, -1)  # (N, D_flat)
            kmeans = KMeans(n_clusters=self.max_size, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(X_feat.cpu().numpy())
            centers = kmeans.cluster_centers_

            keep_indices = []
            # For each cluster pick closest point to its center
            for k in range(self.max_size):
                cluster_inds = np.where(labels == k)[0]
                cluster_points = X_feat[cluster_inds].cpu().numpy()
                dists = np.linalg.norm(cluster_points - centers[k], axis=1)
                if len(dists) == 0:
                    continue  # Empty cluster
                chosen = cluster_inds[np.argmin(dists)]
                keep_indices.append(int(chosen))
            # Optional: stable ordering (by cluster id)
            # (Already in order k = 0..K-1)

        else:
            raise ValueError(f"Unknown data_selection strategy: {self.data_selection}")

        # Rebuild deques with selected samples
        new_x = deque(xs[i] for i in keep_indices)
        new_y = deque(ys[i] for i in keep_indices)

        self.data_x = new_x
        self.data_y = new_y

    def get_dataset(self, as_numpy: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current dataset."""
        X, Y = self.tensors()
        if as_numpy:
            return X.cpu().numpy(), Y.cpu().numpy()
        return X, Y

    def clear(self) -> None:
        """Clear the buffer."""
        self.data_x.clear()
        self.data_y.clear()

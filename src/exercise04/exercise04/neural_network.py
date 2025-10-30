import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

max_params = 1e6


class NeuralNetwork(nn.Module):
    def __init__(self, hyperparameters: dict = None, init_from_checkpoint: bool = False, checkpoint_path: str = None):
        super().__init__()
        self.save_dir = os.path.join(os.path.dirname(__file__), "outputs")
        self.checkpoint_path = (
            os.path.join(self.save_dir, "nn_checkpoint_ex04.ckpt") if checkpoint_path is None else checkpoint_path
        )
        if hyperparameters is None or init_from_checkpoint:
            assert init_from_checkpoint, "Hyperparameters must be provided if not loading from checkpoint."
            self._load_from_checkpoint()
            hyperparameters = self.hyperparameters
        else:
            hyperparameter_keys = hyperparameters.keys()
            assert "input_dim" in hyperparameter_keys and "output_dim" in hyperparameter_keys, (
                "Hyperparameters must contain input_dim, output_dim."
            )
            self.hyperparameters = hyperparameters

        ########################################################################
        # Task 3
        # TODO:
        # 1. Create a network model architecture using nn.[layerTypes]
        # Hints:
        # 1. A simple way to define a sequential network is passing
        # a (dereferenced) list of layers into nn.Sequential
        # 2. Simple fully connected layers (nn.Linear) + non-linear activation
        # functions are sufficient here
        # 3. Assign your network to the variable "network"
        ########################################################################
        # Show relevant variables
        input_dim = hyperparameters["input_dim"]
        output_dim = hyperparameters["output_dim"]
        network = None
        



















        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        self.network = network
        if init_from_checkpoint:
            self.load_state_dict(self.model_state_dict)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Model initialized with {num_params} trainable parameters.")
        assert num_params <= max_params, f"Model must have less than {max_params} parameters."

    def forward(self, x):
        # TODO: Optionally adjust the forward function if required by your architecture
        return self.network(x)

    def _save_checkpoint(self, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()
        # Copy and move all tensors to CPU for saving. This is required as ARTEMIS servers only run on CPU.
        state_dict_cpu = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        checkpoint = {"model_state_dict": state_dict_cpu, "hyperparameters": self.hyperparameters}
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(checkpoint, self.checkpoint_path)

    def _load_from_checkpoint(self):
        # assert os.path.exists(self.save_dir), "Checkpoint directory does not exist."
        # assert os.path.isdir(self.save_dir), "Checkpoint path is not a directory."
        assert os.path.exists(self.checkpoint_path), f"Checkpoint file {self.checkpoint_path} does not exist. Make sure to train a model first and commit the checkpoint to the repository."
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model_state_dict = checkpoint["model_state_dict"]
        self.hyperparameters = checkpoint["hyperparameters"]


class RegressionTrainer:
    def __init__(self, model: NeuralNetwork, cfg: dict = {}, device="cpu"):
        self.model = model
        self.device = device
        self.get_optimizer_criterion_scheduler(cfg)
        self.train_losses = []
        self.val_losses = []

        self.model.to(self.device)

    def get_optimizer_criterion_scheduler(self, cfg):
        ########################################################################
        # Task 4
        # TODO:
        # 1. Create optimizer
        # 2. Create scheduler
        # 3. Create loss function
        # Hints:
        # 1. Use optimizers from torch.optim
        # 2. Use learning rate schedulers from torch.optim.lr_scheduler
        # 3. Use loss functions from torch.nn
        ########################################################################
        # Show relevant variables
        model = self.model
        optimizer_cfg = cfg.get("optimizer", {})
        scheduler_cfg = cfg.get("scheduler", {})
        criterion_cfg = cfg.get("criterion", {})  # noqa: F841
        optimizer, scheduler, criterion = None, None, None # Define those
        ########################################################################
        












        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        self.optimizer, self.scheduler, self.criterion = optimizer, scheduler, criterion

    def train_epoch(self, train_loader):
        ########################################################################
        # Task 4
        # TODO:
        # 1. Set the model on training mode
        # 2. Loop over the training data for one epoch using the train_loader
        # and calculate outputs, losses, update the model weights, and
        # accumulate the total loss
        # Hints:
        # 1. Don't forget to reset the gradients at the start of each iteration
        ########################################################################
        # Show relevant variables
        total_loss = 0
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device
        ########################################################################
        














        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(val_loader) if len(val_loader) > 0 else float("inf")

    def train(self, train_loader, val_loader=None, epochs=100, patience=10):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state_dict = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                    if best_state_dict is not None:
                        self.model.load_state_dict(best_state_dict)
                    break

            if epoch % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        # Save the best model (if validation was used), else save last
        self.model._save_checkpoint(best_state_dict)
        print("Training complete. Checkpoint saved")

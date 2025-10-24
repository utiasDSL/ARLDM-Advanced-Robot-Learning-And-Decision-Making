import fire  # type: ignore
import wandb  # type: ignore
from ml_collections import ConfigDict  # type: ignore

from exercise06 import ppo  # type: ignore

assert False, "This script is currently not being used. It provides a template for conducting wandb sweeps. The usage of ppo.PPOTrainer needs to be checked before using this script. Also requires `pip install fire`"

sweep_config = {
    "method": "random",
    "metric": {"name": "eval/mean_rewards", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "uniform", "min": 1e-4, "max": 5e-3},
        "clip_coef": {"distribution": "uniform", "min": 0.2, "max": 0.3},
        "ent_coef": {"distribution": "uniform", "min": 0.0, "max": 0.05},
        "gamma": {"distribution": "uniform", "min": 0.8, "max": 0.99},
        "gae_lambda": {"distribution": "uniform", "min": 0.9, "max": 0.99},
        "max_grad_norm": {"distribution": "uniform", "min": 1.0, "max": 5.0},
    },
}

config = ConfigDict(
    {
        "n_envs": 1024,
        "device": "cpu",
        "total_timesteps": 1_000_000,
        "learning_rate": 1.5e-3,
        "n_steps": 16,  # Number of steps per environment per policy rollout
        "gamma": 0.90,  # Discount factor
        "gae_lambda": 0.95,  # Lambda for general advantage estimation
        "n_minibatches": 16,  # Number of mini-batches
        "n_epochs": 15,
        "norm_adv": True,
        "clip_coef": 0.25,
        "clip_vloss": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 5.0,
        "target_kl": None,
        "seed": 0,
        "n_eval_envs": 64,
        "n_eval_steps": 1_000,
        "save_model": False,
        "eval_interval": 999_000,
        "lr_decay": True,
    }
)


def main(n_runs: int | None = None, sweep: str | None = None):
    wandb.login()
    project = "ARLDM-Exercise-PPO-HPO"

    if sweep is None:
        sweep = wandb.sweep(sweep_config, project=project)

    wandb.agent(
        sweep,
        lambda: ppo.PPOTrainer(config.copy_and_resolve_references(), True),
        count=n_runs,
        project=project,
    )


if __name__ == "__main__":
    fire.Fire(main)

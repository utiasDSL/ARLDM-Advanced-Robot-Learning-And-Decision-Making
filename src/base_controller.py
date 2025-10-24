from abc import ABC, abstractmethod
from typing import Any

from crazyflow.sim.sim import Sim
from crazyflow.sim.symbolic import SymbolicModel, symbolic_from_sim


class BaseController(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step_control(self):
        pass

    def learn(self, env=None, **kwargs: Any):
        """Performs learning (pre-training, training, fine-tuning, etc).

        Args:
            env (BenchmarkEnv): The environment to be used for training.
            **kwargs: additional keyword arguments.
        """
        return

    def get_symbolic(self, env: Sim) -> SymbolicModel:
        """Fetch the prior model from the env for the controller.

        Args:
            env (BenchmarkEnv): the environment to fetch prior model from.

        Returns:
            SymbolicModel: CasAdi prior model.
        """
        symbolic_model = symbolic_from_sim(env.sim)

        return symbolic_model

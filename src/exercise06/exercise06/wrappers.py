import jax.numpy as jp
from gymnasium.spaces import Box, flatten_space
from gymnasium.vector import VectorActionWrapper, VectorEnv, VectorObservationWrapper


class FlattenJaxObservation(VectorObservationWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        return jp.concatenate([v for v in observations.values()], axis=-1)

class ZeroYaw(VectorActionWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        assert isinstance(env.single_action_space, Box)
        assert env.single_action_space.shape[0] == 4

    def actions(self, actions: jp.ndarray) -> jp.ndarray:
        # Simply set the yaw action (4th dimension) to zero
        # Omitting the yaw command from policy output
        return actions.at[..., 3].set(0.0)
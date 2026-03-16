from typing import TypeVar

import jax.numpy as jnp
import numpy as np
from gymnasium.vector import VectorEnv, VectorObservationWrapper

ArrayType = TypeVar("ArrayType")


class NoisyObservationWrapper(VectorObservationWrapper):
    def __init__(self, env: VectorEnv, noise_std=0.01):
        """Adds Gaussian noise to the observation.

        :param env: The Gym environment to be wrapped.
        :param noise_std: Standard deviation of the Gaussian noise.
        """
        super().__init__(env)
        self.noise_std = noise_std  # Controls the noise intensity

    def observations(self, observations):
        """Applies noise to each numerical entry in the observation dictionary.

        :param obs: The original observation from the environment.
        :return: The observation with added noise.
        """
        noisy_obs = {}
        for key, value in observations.items():
            # numpy array
            if isinstance(value, np.ndarray):
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=value.shape)
                noisy_obs[key] = value + noise  # Add Gaussian noise
            # JAX array, jaxlib.xla_extension.ArrayImpl
            elif "jaxlib.xla_extension.ArrayImpl" in str(type(value)):
                # convert to numpy array
                value_np = np.array(value)
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=value_np.shape)
                noisy_value = value_np + noise
                # convert it back
                noisy_obs[key] = jnp.array(noisy_value)
            else:
                # print(f"Key '{key}': unhandled type {type(value)}")
                noisy_obs[key] = value  # Keep non-numerical or other values unchanged
        return noisy_obs

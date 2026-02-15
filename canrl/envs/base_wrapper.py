"""Base wrapper class with common functionality."""

from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np


class BaseWrapper(gym.Wrapper):
    """
    Base wrapper providing common utilities for custom wrappers.
    
    Inherit from this class when creating new wrappers.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_reward = 0.0
        self._episode_length = 0
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and internal counters."""
        self._episode_reward = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)
    
    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment and track episode stats."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(reward)
        self._episode_length += 1
        return obs, reward, terminated, truncated, info
    
    @property
    def episode_reward(self) -> float:
        """Current episode cumulative reward."""
        return self._episode_reward
    
    @property
    def episode_length(self) -> int:
        """Current episode length."""
        return self._episode_length

"""Frame stacking wrapper for temporal information."""

from typing import Any
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FrameStack(gym.Wrapper):
    """
    Stack consecutive frames for temporal information.
    
    Useful for visual observations where motion information is important
    (e.g., Atari games, robotics with camera input).
    
    Args:
        env: The environment to wrap.
        num_stack: Number of frames to stack.
        
    Example:
        >>> env = gym.make("CartPole-v1")
        >>> env = FrameStack(env, num_stack=4)
        >>> obs, _ = env.reset()
        >>> obs.shape  # (4, original_obs_shape...)
    """
    
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self._frames: deque = deque(maxlen=num_stack)
        
        # Update observation space
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and initialize frame stack with copies of first observation."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self._frames.append(obs)
        return self._get_observation(), info
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Stack frames into single observation."""
        return np.array(self._frames)

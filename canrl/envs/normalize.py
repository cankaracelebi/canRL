"""Observation and reward normalization wrappers."""

from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and standard deviation using Welford's algorithm.
    
    Numerically stable online computation of mean and variance.
    """
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Avoid division by zero
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        self.var = m2 / total_count
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        """Standard deviation."""
        return np.sqrt(self.var + self.epsilon)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics."""
        return (x - self.mean) / self.std
    
    def save(self) -> dict:
        """Save statistics to dictionary."""
        return {"mean": self.mean, "var": self.var, "count": self.count}
    
    def load(self, stats: dict) -> None:
        """Load statistics from dictionary."""
        self.mean = stats["mean"]
        self.var = stats["var"]
        self.count = stats["count"]


class NormalizeObservation(gym.Wrapper):
    """
    Normalize observations using running mean and standard deviation.
    
    Args:
        env: The environment to wrap.
        epsilon: Small constant for numerical stability.
        clip: Clip normalized observations to [-clip, clip].
        update_stats: Whether to update running statistics during stepping.
        
    Example:
        >>> env = gym.make("Pendulum-v1")
        >>> env = NormalizeObservation(env)
        >>> obs, _ = env.reset()  # Observations now normalized
    """
    
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        update_stats: bool = True,
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self.update_stats = update_stats
        
        obs_shape = env.observation_space.shape
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        
        # Update observation space to normalized range
        self.observation_space = spaces.Box(
            low=-clip, high=clip, shape=obs_shape, dtype=np.float32
        )
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and normalize initial observation."""
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info
    
    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step and normalize observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_obs(obs), reward, terminated, truncated, info
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        if self.update_stats:
            self.obs_rms.update(obs[np.newaxis, ...])
        
        normalized = self.obs_rms.normalize(obs)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
    
    def set_update_stats(self, update: bool) -> None:
        """Enable/disable statistics updates (useful for evaluation)."""
        self.update_stats = update


class NormalizeReward(gym.Wrapper):
    """
    Normalize rewards using running standard deviation.
    
    Uses return-based normalization for more stable training.
    
    Args:
        env: The environment to wrap.
        gamma: Discount factor for return computation.
        epsilon: Small constant for numerical stability.
        clip: Clip normalized rewards to [-clip, clip].
        
    Example:
        >>> env = gym.make("Pendulum-v1")
        >>> env = NormalizeReward(env, gamma=0.99)
    """
    
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        
        self.return_rms = RunningMeanStd(shape=())
        self._returns = 0.0
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and clear return accumulator."""
        self._returns = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step and normalize reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track discounted returns for normalization
        self._returns = self._returns * self.gamma + reward
        self.return_rms.update(np.array([self._returns]))
        
        # Normalize by return std
        normalized_reward = reward / (self.return_rms.std + self.epsilon)
        normalized_reward = np.clip(normalized_reward, -self.clip, self.clip)
        
        if terminated or truncated:
            self._returns = 0.0
        
        return obs, float(normalized_reward), terminated, truncated, info

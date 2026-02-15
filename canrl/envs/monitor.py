"""Episode monitoring wrapper."""

from typing import Any, SupportsFloat
import time
import gymnasium as gym
import numpy as np


class Monitor(gym.Wrapper):
    """
    Track episode statistics (returns, lengths, timing).
    
    Records episode results and provides access to history
    for logging and analysis.
    
    Args:
        env: The environment to wrap.
        
    Attributes:
        episode_returns: List of episode total rewards.
        episode_lengths: List of episode step counts.
        episode_times: List of episode durations in seconds.
        
    Example:
        >>> env = gym.make("CartPole-v1")
        >>> env = Monitor(env)
        >>> obs, _ = env.reset()
        >>> # ... run episode ...
        >>> print(env.episode_returns[-1])  # Last episode return
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Episode history
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        
        # Current episode tracking
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_start_time = 0.0
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and start tracking new episode."""
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_start_time = time.time()
        
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step environment and update episode statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._episode_return += float(reward)
        self._episode_length += 1
        
        # Record episode on completion
        if terminated or truncated:
            episode_time = time.time() - self._episode_start_time
            
            self.episode_returns.append(self._episode_return)
            self.episode_lengths.append(self._episode_length)
            self.episode_times.append(episode_time)
            
            # Add episode info to info dict
            info["episode"] = {
                "return": self._episode_return,
                "length": self._episode_length,
                "time": episode_time,
            }
        
        return obs, reward, terminated, truncated, info
    
    @property
    def total_episodes(self) -> int:
        """Total number of completed episodes."""
        return len(self.episode_returns)
    
    @property
    def total_steps(self) -> int:
        """Total steps across all completed episodes."""
        return sum(self.episode_lengths)
    
    def get_statistics(self) -> dict[str, float]:
        """
        Get summary statistics for completed episodes.
        
        Returns:
            Dictionary with mean/std of returns, lengths, and times.
        """
        if not self.episode_returns:
            return {}
        
        returns = np.array(self.episode_returns)
        lengths = np.array(self.episode_lengths)
        
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_length": float(np.mean(lengths)),
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }
    
    def get_recent_statistics(self, n: int = 100) -> dict[str, float]:
        """Get statistics for last n episodes."""
        if not self.episode_returns:
            return {}
        
        recent_returns = np.array(self.episode_returns[-n:])
        recent_lengths = np.array(self.episode_lengths[-n:])
        
        return {
            "mean_return": float(np.mean(recent_returns)),
            "std_return": float(np.std(recent_returns)),
            "min_return": float(np.min(recent_returns)),
            "max_return": float(np.max(recent_returns)),
            "mean_length": float(np.mean(recent_lengths)),
            "num_episodes": len(recent_returns),
        }

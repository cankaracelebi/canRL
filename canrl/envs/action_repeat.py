"""Action repeat wrapper for frame skipping."""

from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    """
    Repeat each action for multiple steps (frame skipping).
    
    Reduces the effective frequency of decision making while
    accumulating rewards over repeated steps.
    
    Args:
        env: The environment to wrap.
        repeat: Number of times to repeat each action.
        
    Example:
        >>> env = gym.make("CartPole-v1")
        >>> env = ActionRepeat(env, repeat=4)
        >>> # Each action is now executed 4 times
    """
    
    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        assert repeat >= 1, "Repeat must be >= 1"
        self.repeat = repeat
    
    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute action repeatedly, accumulating rewards.
        
        Returns the final observation and cumulative reward.
        Terminates early if episode ends.
        """
        total_reward = 0.0
        
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info

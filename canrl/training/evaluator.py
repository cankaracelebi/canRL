"""Evaluation utilities for RL agents."""

from typing import Any
import gymnasium as gym
import numpy as np

from canrl.algorithms.base import BaseAgent


class Evaluator:
    """
    Evaluate an RL agent's performance.
    
    Runs the agent in deterministic mode and collects
    episode statistics.
    
    Args:
        env: Evaluation environment.
        num_episodes: Number of episodes per evaluation.
        max_steps: Maximum steps per episode (None for env default).
        render: Whether to render during evaluation.
        
    Example:
        >>> evaluator = Evaluator(
        ...     env=gym.make("CartPole-v1"),
        ...     num_episodes=10,
        ... )
        >>> stats = evaluator.evaluate(agent)
        >>> print(f"Mean return: {stats['mean_return']:.2f}")
    """
    
    def __init__(
        self,
        env: gym.Env,
        num_episodes: int = 10,
        max_steps: int | None = None,
        render: bool = False,
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.render = render
    
    def evaluate(self, agent: BaseAgent) -> dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            agent: Agent to evaluate.
            
        Returns:
            Dictionary of evaluation statistics.
        """
        returns = []
        lengths = []
        
        for _ in range(self.num_episodes):
            episode_return, episode_length = self._run_episode(agent)
            returns.append(episode_return)
            lengths.append(episode_length)
        
        returns = np.array(returns)
        lengths = np.array(lengths)
        
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_length": float(np.mean(lengths)),
        }
    
    def _run_episode(self, agent: BaseAgent) -> tuple[float, int]:
        """Run a single evaluation episode."""
        state, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        
        while True:
            if self.render:
                self.env.render()
            
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = self.env.step(action)
            
            episode_return += reward
            episode_length += 1
            
            if terminated or truncated:
                break
            
            if self.max_steps and episode_length >= self.max_steps:
                break
        
        return episode_return, episode_length
    
    def evaluate_single(self, agent: BaseAgent) -> tuple[float, int, list[np.ndarray]]:
        """
        Run single episode and return trajectory.
        
        Useful for visualization and debugging.
        
        Returns:
            (episode_return, episode_length, list_of_states)
        """
        states = []
        state, _ = self.env.reset()
        states.append(state.copy())
        
        episode_return = 0.0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = self.env.step(action)
            
            states.append(state.copy())
            episode_return += reward
            episode_length += 1
            
            if terminated or truncated:
                break
            
            if self.max_steps and episode_length >= self.max_steps:
                break
        
        return episode_return, episode_length, states

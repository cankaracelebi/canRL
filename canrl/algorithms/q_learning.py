"""
Tabular Q-Learning Agent Skeleton.

TODO: Implement your Q-Learning algorithm here!

Q-Learning is a model-free, off-policy TD control algorithm.
Update rule: Q(s,a) <- Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))

This skeleton provides the structure - you fill in the logic.
"""

from typing import Any
from pathlib import Path
import numpy as np
import pickle

from canrl.agents.base_agent import BaseAgent
from canrl.buffers.base_buffer import Batch


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning Agent.
    
    For discrete state and action spaces only.
    
    TODO: Implement the Q-learning algorithm.
    
    Example usage (after you implement):
        >>> agent = QLearningAgent(
        ...     state_dim=16,  # e.g., FrozenLake 4x4
        ...     action_dim=4,
        ...     learning_rate=0.1,
        ...     gamma=0.99,
        ... )
        >>> action = agent.select_action(state)
        >>> agent.update_step(state, action, reward, next_state, done)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_dim: Number of discrete states.
            action_dim: Number of discrete actions.
            learning_rate: Learning rate (alpha).
            gamma: Discount factor.
            epsilon_*: Exploration parameters.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # TODO: Initialize Q-table
        # Hint: self.q_table = np.zeros((state_dim, action_dim))
        self.q_table = None  # YOUR CODE HERE
        
        raise NotImplementedError(
            "QLearningAgent not implemented yet! "
            "Edit this file to add your implementation."
        )
    
    def select_action(self, state: np.ndarray | int, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        TODO: Implement epsilon-greedy action selection.
        
        Args:
            state: Current state (discrete index or array).
            deterministic: If True, always select greedy action.
            
        Returns:
            Selected action index.
        """
        # YOUR CODE HERE
        # Hint:
        # if not deterministic and np.random.random() < self.epsilon:
        #     return np.random.randint(self.action_dim)
        # return np.argmax(self.q_table[state])
        raise NotImplementedError()
    
    def update(self, batch: Batch) -> dict[str, float]:
        """
        Update Q-table using a batch (for compatibility with Trainer).
        
        Note: For true tabular Q-learning, you'd typically call
        update_step() after each environment step instead.
        """
        # TODO: Implement batch update if needed
        raise NotImplementedError()
    
    def update_step(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> dict[str, float]:
        """
        Single-step Q-learning update.
        
        TODO: Implement the Q-learning update rule.
        
        Q(s,a) <- Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        
        Returns:
            Dictionary with update metrics.
        """
        # YOUR CODE HERE
        # Hint:
        # if done:
        #     target = reward
        # else:
        #     target = reward + self.gamma * np.max(self.q_table[next_state])
        # 
        # td_error = target - self.q_table[state, action]
        # self.q_table[state, action] += self.learning_rate * td_error
        # 
        # # Decay epsilon
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # 
        # return {"td_error": abs(td_error), "epsilon": self.epsilon}
        raise NotImplementedError()
    
    def save(self, path: str | Path) -> None:
        """Save Q-table and parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "config": self.get_config(),
            }, f)
    
    def load(self, path: str | Path) -> None:
        """Load Q-table and parameters."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_end)
    
    def get_config(self) -> dict[str, Any]:
        """Return agent configuration."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

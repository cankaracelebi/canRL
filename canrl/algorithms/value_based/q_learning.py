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

from canrl.algorithms.base import BaseAgent
from canrl.buffers.base_buffer import Batch


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning Agent.
    
    For discrete state and action spaces only.
    

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
        

        self.q_table = np.zeros((state_dim, action_dim)) # state_dim x action dim
        
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
        if deterministic or (self.epsilon == 0.0):
            return np.argmax(self.q_table[state])
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        
        else:
            action = np.argmax(self.q_table[state])

        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)

        return action
    
    def update(self, batch: Batch) -> dict[str, float]:
        """
        Update Q-table using a batch (for compatibility with Trainer).
        # batch update basically non existing for tabular q -> no buffers no neural nets!
        # TODO: Implement batch update if needed
        """
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
        
        
        Q(s,a) <- Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        
        Returns:
            Dictionary with update metrics.
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])
        
        error  = target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * error


        return {"error": error} # what else?
    
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

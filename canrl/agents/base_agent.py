"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    All agent implementations should inherit from this class
    and implement the required methods.
    
    Your algorithm implementations (Q-Learning, DQN, etc.) should
    inherit from this class.
    
    Example:
        class DQNAgent(BaseAgent):
            def __init__(self, state_dim, action_dim, ...):
                self.q_network = ...
                self.target_network = ...
                
            def select_action(self, state, deterministic=False):
                if not deterministic and np.random.random() < self.epsilon:
                    return np.random.randint(self.action_dim)
                return self.q_network(state).argmax()
                
            def update(self, batch):
                # Compute TD loss and update Q-network
                ...
                return {"loss": loss, "q_mean": q_values.mean()}
    """
    
    @abstractmethod
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray | int:
        """
        Select an action given the current state.
        
        Args:
            state: Current observation/state.
            deterministic: If True, select greedy action (no exploration).
                          If False, may include exploration (epsilon-greedy, etc.)
        
        Returns:
            The selected action (int for discrete, array for continuous).
        """
        pass
    
    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """
        Update the agent using a batch of experience.
        
        Args:
            batch: Batch of transitions (from replay buffer).
        
        Returns:
            Dictionary of training metrics (loss, Q-values, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save agent state to disk.
        
        Should save all necessary components to resume training
        or run inference (networks, optimizers, hyperparameters).
        
        Args:
            path: Directory or file path to save to.
        """
        pass
    
    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: Directory or file path to load from.
        """
        pass
    
    def train_mode(self) -> None:
        """Set agent to training mode (enables exploration, dropout, etc.)."""
        pass
    
    def eval_mode(self) -> None:
        """Set agent to evaluation mode (disables exploration, dropout, etc.)."""
        pass
    
    def get_config(self) -> dict[str, Any]:
        """
        Return agent configuration for logging/checkpointing.
        
        Override to include your algorithm's hyperparameters.
        """
        return {}

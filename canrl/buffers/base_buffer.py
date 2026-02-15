"""Abstract base class for replay buffers."""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple
import numpy as np


class Transition(NamedTuple):
    """A single transition (s, a, r, s', done)."""
    
    state: np.ndarray
    action: np.ndarray | int
    reward: float
    next_state: np.ndarray
    done: bool
    
    # Optional fields for algorithms that need them
    info: dict[str, Any] | None = None


class Batch(NamedTuple):
    """A batch of transitions for training."""
    
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    
    # For prioritized replay
    weights: np.ndarray | None = None
    indices: np.ndarray | None = None


class BaseBuffer(ABC):
    """
    Abstract base class for experience replay buffers.
    
    Implement this interface to create custom buffer types.
    
    Args:
        capacity: Maximum number of transitions to store.
        state_shape: Shape of state observations.
        action_shape: Shape of actions (empty tuple for discrete).
        
    Example:
        class MyCustomBuffer(BaseBuffer):
            def add(self, transition):
                # Custom storage logic
                pass
                
            def sample(self, batch_size):
                # Custom sampling logic
                pass
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
    
    @abstractmethod
    def add(self, transition: Transition) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            transition: The transition to store.
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Batch of transitions.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return current number of stored transitions."""
        pass
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions to sample."""
        return len(self) >= batch_size
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self) >= self.capacity

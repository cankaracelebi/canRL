"""Standard uniform replay buffer."""

import numpy as np
from canrl.buffers.base_buffer import BaseBuffer, Transition, Batch


class ReplayBuffer(BaseBuffer):
    """
    Standard replay buffer with uniform random sampling.
    
    Stores transitions in numpy arrays for efficient memory usage
    and fast sampling.
    
    Args:
        capacity: Maximum number of transitions to store.
        state_shape: Shape of state observations.
        action_shape: Shape of actions (empty tuple for discrete).
        
    Example:
        >>> buffer = ReplayBuffer(10000, state_shape=(4,))
        >>> buffer.add(Transition(state, action, reward, next_state, done))
        >>> batch = buffer.sample(32)
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
    ):
        super().__init__(capacity, state_shape, action_shape)
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self._position = 0
        self._size = 0
    
    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer (circular)."""
        self.states[self._position] = transition.state
        self.actions[self._position] = transition.action
        self.rewards[self._position] = transition.reward
        self.next_states[self._position] = transition.next_state
        self.dones[self._position] = transition.done
        
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add multiple transitions at once for efficiency."""
        batch_size = len(states)
        
        if self._position + batch_size <= self.capacity:
            # Can fit without wrapping
            end = self._position + batch_size
            self.states[self._position:end] = states
            self.actions[self._position:end] = actions
            self.rewards[self._position:end] = rewards
            self.next_states[self._position:end] = next_states
            self.dones[self._position:end] = dones
        else:
            # Need to wrap around
            first_part = self.capacity - self._position
            self.states[self._position:] = states[:first_part]
            self.actions[self._position:] = actions[:first_part]
            self.rewards[self._position:] = rewards[:first_part]
            self.next_states[self._position:] = next_states[:first_part]
            self.dones[self._position:] = dones[:first_part]
            
            second_part = batch_size - first_part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]
        
        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)
    
    def sample(self, batch_size: int) -> Batch:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self._size, size=batch_size)
        
        return Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            weights=None,
            indices=indices,
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size
    
    def clear(self) -> None:
        """Clear all stored transitions."""
        self._position = 0
        self._size = 0

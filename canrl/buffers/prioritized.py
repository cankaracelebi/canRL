"""Prioritized Experience Replay (PER) buffer."""

import numpy as np
from canrl.buffers.base_buffer import BaseBuffer, Transition, Batch


class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.
    
    Enables O(log n) sampling proportional to priorities
    and O(log n) priority updates.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
    
    def add(self, priority: float) -> int:
        """Add a new priority and return its data index."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return data_idx
    
    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority at given tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get(self, value: float) -> tuple[int, float, int]:
        """
        Get leaf based on cumulative priority value.
        
        Returns:
            (tree_idx, priority, data_idx)
        """
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            if left_idx >= len(self.tree):
                # Reached leaf
                break
            
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = parent_idx - self.capacity + 1
        return parent_idx, self.tree[parent_idx], data_idx
    
    @property
    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]
    
    @property
    def max_priority(self) -> float:
        """Maximum priority in tree."""
        return np.max(self.tree[-self.capacity:])


class PrioritizedReplayBuffer(BaseBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions proportional to their TD-error priority,
    with importance sampling weights to correct for bias.
    
    Reference: "Prioritized Experience Replay" (Schaul et al., 2015)
    
    Args:
        capacity: Maximum number of transitions.
        state_shape: Shape of state observations.
        action_shape: Shape of actions.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta: Initial importance sampling exponent.
        beta_increment: Amount to increase beta per sample call.
        epsilon: Small constant added to priorities.
        
    Example:
        >>> buffer = PrioritizedReplayBuffer(10000, state_shape=(4,), alpha=0.6)
        >>> buffer.add(Transition(state, action, reward, next_state, done))
        >>> batch = buffer.sample(32)
        >>> # Use batch.weights for importance sampling
        >>> # Update priorities with TD-errors:
        >>> buffer.update_priorities(batch.indices, td_errors)
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        super().__init__(capacity, state_shape, action_shape)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Sum tree for prioritized sampling
        self.tree = SumTree(capacity)
        
        # Storage arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self._size = 0
        self._max_priority = 1.0
    
    def add(self, transition: Transition) -> None:
        """Add transition with maximum priority."""
        data_idx = self.tree.add(self._max_priority ** self.alpha)
        
        self.states[data_idx] = transition.state
        self.actions[data_idx] = transition.action
        self.rewards[data_idx] = transition.reward
        self.next_states[data_idx] = transition.next_state
        self.dones[data_idx] = transition.done
        
        self._size = min(self._size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Batch:
        """Sample batch with priority-weighted probabilities."""
        indices = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Stratified sampling for lower variance
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            
            tree_idx, priority, data_idx = self.tree.get(value)
            tree_indices[i] = tree_idx
            priorities[i] = priority
            indices[i] = data_idx
        
        # Compute importance sampling weights
        probs = priorities / self.tree.total
        min_prob = np.min(probs)
        
        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # IS weights: (N * P(i))^(-beta) / max_weight
        weights = (self._size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize by max for stability
        
        return Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            weights=weights.astype(np.float32),
            indices=tree_indices,  # Tree indices for priority updates
        )
    
    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD-errors.
        
        Args:
            tree_indices: Tree indices from batch.indices
            td_errors: Absolute TD-errors for each transition
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), priority)
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size

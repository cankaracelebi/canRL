"""
Template for implementing new RL algorithms.

Copy this file and use it as a starting point for your algorithms.

Usage:
    1. Copy this file: cp _template.py my_algorithm.py
    2. Rename the class
    3. Implement the TODO sections
    4. Add to algorithms/__init__.py
"""

from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from canrl.agents.base_agent import BaseAgent
from canrl.buffers.base_buffer import Batch
from canrl.networks.mlp import MLP
from canrl.utils.schedule import LinearSchedule


class TemplateAgent(BaseAgent):
    """
    Template agent - copy and modify for your implementation.
    
    This shows the basic structure of an agent implementation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        device: str = "auto",
    ):
        """
        Initialize the agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            hidden_dims: Hidden layer sizes.
            learning_rate: Optimizer learning rate.
            gamma: Discount factor.
            epsilon_*: Exploration schedule parameters.
            device: Device to use ('cpu', 'cuda', or 'auto').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # TODO: Create your networks
        # Example: Q-network
        self.network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        # TODO: Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Exploration schedule
        self.epsilon_schedule = LinearSchedule(
            epsilon_start, epsilon_end, epsilon_decay_steps
        )
        self._step = 0
        self._training = True
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        TODO: Implement your action selection logic.
        """
        epsilon = 0.0 if deterministic else self.epsilon_schedule(self._step)
        
        if np.random.random() < epsilon:
            # Random action
            return np.random.randint(self.action_dim)
        
        # Greedy action
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_t)
            return q_values.argmax(dim=1).item()
    
    def update(self, batch: Batch) -> dict[str, float]:
        """
        Update the agent using a batch of experience.
        
        TODO: Implement your learning algorithm.
        
        Returns dict of metrics for logging.
        """
        self._step += 1
        
        # Convert batch to tensors
        states = torch.FloatTensor(batch.states).to(self.device)
        actions = torch.LongTensor(batch.actions).to(self.device)
        rewards = torch.FloatTensor(batch.rewards).to(self.device)
        next_states = torch.FloatTensor(batch.next_states).to(self.device)
        dones = torch.FloatTensor(batch.dones).to(self.device)
        
        # TODO: Compute loss
        # Example: Q-learning target
        with torch.no_grad():
            next_q_values = self.network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * (1 - dones) * max_next_q
        
        # Current Q-values
        current_q = self.network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Loss
        loss = nn.functional.mse_loss(current_q, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "epsilon": self.epsilon_schedule(self._step),
        }
    
    def save(self, path: str | Path) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "config": self.get_config(),
        }, path)
    
    def load(self, path: str | Path) -> None:
        """Load agent state."""
        data = torch.load(path, map_location=self.device)
        self.network.load_state_dict(data["network"])
        self.optimizer.load_state_dict(data["optimizer"])
        self._step = data.get("step", 0)
    
    def train_mode(self) -> None:
        """Set to training mode."""
        self._training = True
        self.network.train()
    
    def eval_mode(self) -> None:
        """Set to evaluation mode."""
        self._training = False
        self.network.eval()
    
    def get_config(self) -> dict[str, Any]:
        """Return agent configuration."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "device": str(self.device),
        }

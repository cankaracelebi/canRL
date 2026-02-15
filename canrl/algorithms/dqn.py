"""
Deep Q-Network (DQN) Agent Skeleton.

TODO: Implement your DQN algorithm here!

DQN uses a neural network to approximate Q-values with:
- Experience replay for sample efficiency
- Target network for training stability

Reference: "Playing Atari with Deep RL" (Mnih et al., 2013)

This skeleton provides the structure - you fill in the logic.
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


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent.
    
    TODO: Implement the DQN algorithm.
    
    Key components to implement:
    1. Q-network and target network
    2. Epsilon-greedy action selection
    3. TD loss computation
    4. Target network updates (soft or hard)
    
    Example usage (after you implement):
        >>> agent = DQNAgent(
        ...     state_dim=4,
        ...     action_dim=2,
        ...     hidden_dims=(64, 64),
        ... )
        >>> action = agent.select_action(state)
        >>> metrics = agent.update(batch)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        target_update_frequency: int = 1000,
        tau: float = 1.0,  # 1.0 = hard update, <1.0 = soft update
        device: str = "auto",
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            hidden_dims: Hidden layer sizes for Q-network.
            learning_rate: Optimizer learning rate.
            gamma: Discount factor.
            epsilon_*: Exploration schedule parameters.
            target_update_frequency: Steps between target network updates.
            tau: Target network update coefficient.
            device: Device to use.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        
        # Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Q-network (Q)
        self.q_network = None  # YOUR CODE HERE
        self.q_network = MLP(state_dim, action_dim, hidden_dims,  nn.RELU).to(device)
        
        #  Create target network (copy of Q-network) (y_t)

        self.target_network = MLP(state_dim, action_dim, hidden_dims, nn.RELU).to(device)
        self.target_network.load_state_dict(self.q_network)
        
        # TODO: Create optimizer
        # Hint: self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.optimizer = None  # YOUR CODE HERE
        self.optimizer = nn.optim.Adam(self.q_network.parameters(), lr=x)
        
        # Exploration schedule
        self.epsilon_schedule = LinearSchedule(
            epsilon_start, epsilon_end, epsilon_decay_steps
        )
        
        self._step = 0
        self._training = True
        
        raise NotImplementedError(
            "DQNAgent not implemented yet! "
            "Edit this file to add your implementation."
        )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        TODO: Implement action selection.
        
        Args:
            state: Current state observation.
            deterministic: If True, always select greedy action.
            
        Returns:
            Selected action index.
        """
        # YOUR CODE HERE
        # Hint:
        # epsilon = 0.0 if deterministic else self.epsilon_schedule(self._step)
        # 
        # if np.random.random() < epsilon:
        #     return np.random.randint(self.action_dim)
        # 
        # with torch.no_grad():
        #     state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #     q_values = self.q_network(state_t)
        #     return q_values.argmax(dim=1).item()
        raise NotImplementedError()
    
    def update(self, batch: Batch) -> dict[str, float]:
        """
        Update Q-network using a batch of experience.
        
        TODO: Implement the DQN update.
        
        Steps:
        1. Compute current Q-values: Q(s, a)
        2. Compute target: r + γ * max_a' Q_target(s', a')
        3. Compute TD loss
        4. Backprop and optimize
        5. Periodically update target network
        
        Returns:
            Dictionary with training metrics.
        """
        # YOUR CODE HERE
        # Hint:
        # self._step += 1
        # 
        # # Convert to tensors
        # states = torch.FloatTensor(batch.states).to(self.device)
        # actions = torch.LongTensor(batch.actions).to(self.device)
        # rewards = torch.FloatTensor(batch.rewards).to(self.device)
        # next_states = torch.FloatTensor(batch.next_states).to(self.device)
        # dones = torch.FloatTensor(batch.dones).to(self.device)
        # 
        # # Current Q-values
        # current_q = self.q_network(states)
        # current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 
        # # Target Q-values
        # with torch.no_grad():
        #     next_q = self.target_network(next_states)
        #     max_next_q = next_q.max(dim=1)[0]
        #     targets = rewards + self.gamma * (1 - dones) * max_next_q
        # 
        # # Loss
        # loss = nn.functional.mse_loss(current_q, targets)
        # 
        # # Optimize
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # 
        # # Update target network
        # if self._step % self.target_update_frequency == 0:
        #     self._update_target_network()
        # 
        # return {
        #     "loss": loss.item(),
        #     "q_mean": current_q.mean().item(),
        #     "epsilon": self.epsilon_schedule(self._step),
        # }
        raise NotImplementedError()
    
    def _update_target_network(self) -> None:
        """
        Update target network.
        
        TODO: Implement target network update.
        
        For hard update (tau=1.0):
            target.load_state_dict(q_network.state_dict())
            
        For soft update (tau<1.0):
            target_param = tau * q_param + (1-tau) * target_param
        """
        # YOUR CODE HERE
        # Hint:
        # if self.tau == 1.0:
        #     self.target_network.load_state_dict(self.q_network.state_dict())
        # else:
        #     for target_param, q_param in zip(
        #         self.target_network.parameters(),
        #         self.q_network.parameters()
        #     ):
        #         target_param.data.copy_(
        #             self.tau * q_param.data + (1 - self.tau) * target_param.data
        #         )
        raise NotImplementedError()
    
    def save(self, path: str | Path) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "config": self.get_config(),
        }, path)
    
    def load(self, path: str | Path) -> None:
        """Load agent state."""
        data = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(data["q_network"])
        self.target_network.load_state_dict(data["target_network"])
        self.optimizer.load_state_dict(data["optimizer"])
        self._step = data.get("step", 0)
    
    def train_mode(self) -> None:
        """Set to training mode."""
        self._training = True
        self.q_network.train()
        self.target_network.train()
    
    def eval_mode(self) -> None:
        """Set to evaluation mode."""
        self._training = False
        self.q_network.eval()
        self.target_network.eval()
    
    def get_config(self) -> dict[str, Any]:
        """Return agent configuration."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_update_frequency": self.target_update_frequency,
            "device": str(self.device),
        }

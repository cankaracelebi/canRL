"""
Deep Q-Network (DQN) Agent


DQN uses a neural network to approximate Q-values with:
- Experience replay for sample efficiency
- Target network for training stability

Reference: "Playing Atari with Deep RL" (Mnih et al., 2013)

"""

from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from canrl.algorithms.base import BaseAgent
from canrl.buffers.base_buffer import Batch
from canrl.networks.mlp import MLP
from canrl.utils.schedule import LinearSchedule


class DQN(BaseAgent):
    """
    Deep Q-Network Agent.


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
        self.q_network = MLP(state_dim, action_dim, hidden_dims,  nn.ReLU).to(device)
        
        #  Create target network (copy of Q-network for y_t)

        self.target_network = MLP(state_dim, action_dim, hidden_dims, nn.ReLU).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # wont be updated via backprop
        
        # used Adam as a generalized practice can be parameterized in the future
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Exploration schedule epsilon greedy with linear decay as a simple baseline
        self.epsilon_schedule = LinearSchedule(
            epsilon_start, epsilon_end, epsilon_decay_steps
        )
        
        self._step = 0
        self._target_update = 0
        self._training = True
        

    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
                
        Args:
            state: Current state observation.
            deterministic: if True, always select greedy action.
            
        Returns:
            Selected action index.
        """
        epsilon = 0.0  if deterministic else self.epsilon_schedule(self._step)
        if np.random.random() < epsilon:
            # if random exp is triggered by the random var -> sample from uniform random distrbution
            return np.random.randint(self.action_dim)

        #otherwise get q vals -> argmax 
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self._step)
            q_vals = self.q_network(state_t)
            return q_vals.argmax(dim=1).item()
        
    
    def update(self, batch: Batch) -> dict[str, float]:
        """
        Update Q-network using a batch of experience.
                
        Steps:
        1. Compute current Q-values: Q(s, a)
        2. Compute target: r + γ * max_a' Q_target(s', a') based on s' being terminal!
        3. Compute TD loss
        4. Backprop and optimize
        5. Periodically update target network  
        Returns:
            Dictionary with training metrics.
        """
        # 
        # ill get target as that and the q vals and use the batch backprop 

        criterion = nn.MSELoss()
        
        batch_size =  batch.states.size 
        self._step += batch_size
        self._target_update += batch_size

        targets = torch.Tensor([])
        q_s = self.q_network(batch.states)
        
        for j in range(batch_size):
            #TODO: convert this to batched torch operation with lambda func

            next_state = batch.next_states[j]
            reward = batch.rewards[j]
            terminal = batch.dones[j]

            if(terminal):
                torch.cat((targets, reward), 0)
            else:

                target = reward  + self.gamma * max(self.target_network(next_state)) # target network is in eval mode detach not needed!
                torch.cat((targets, target), 0)
        print(targets, q_s)
        assert(targets.shape == q_s.shape) # this should work !
        loss = criterion(q_s, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if self._target_update >= self.target_update_frequency:
            self._update_target_network()
            self._target_update -= self.target_update_frequency

        return {"q_loss": loss} 


        
        
    def _update_target_network(self) -> None:
        """
        Update target network.
        
        """

        if self.tau == 1.0:
            #hard update nice and easy just copy the params
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # keep 1- tau portion of the current network 
            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy(self.tau* q_param.data + (1-self.tau) * target_param.data)
    
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

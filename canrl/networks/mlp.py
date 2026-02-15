"""Multi-layer perceptron (MLP) networks."""

from typing import Sequence
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Configurable multi-layer perceptron.
    
    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        hidden_dims: Sequence of hidden layer sizes.
        activation: Activation function class.
        output_activation: Optional activation for output layer.
        
    Example:
        >>> # Q-network for CartPole
        >>> q_net = MLP(
        ...     input_dim=4,
        ...     output_dim=2,
        ...     hidden_dims=[64, 64],
        ... )
        >>> q_values = q_net(torch.randn(32, 4))  # (32, 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] | None = None,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class DuelingMLP(nn.Module):
    """
    Dueling network architecture (for DQN variants).
    
    Separates state value and advantage estimation for
    more stable learning.
    
    Reference: "Dueling Network Architectures" (Wang et al., 2016)
    
    Args:
        input_dim: Input feature dimension.
        action_dim: Number of actions.
        hidden_dims: Sequence of shared hidden layer sizes.
        value_dims: Sequence of value stream hidden sizes.
        advantage_dims: Sequence of advantage stream hidden sizes.
        
    Example:
        >>> net = DuelingMLP(input_dim=4, action_dim=2)
        >>> q_values = net(state)  # Returns Q(s, a) for all actions
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256,),
        value_dims: Sequence[int] = (128,),
        advantage_dims: Sequence[int] = (128,),
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = MLP(
            input_dim=input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
        )
        
        # Value stream: V(s)
        self.value_stream = MLP(
            input_dim=hidden_dims[-1],
            output_dim=1,
            hidden_dims=value_dims,
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = MLP(
            input_dim=hidden_dims[-1],
            output_dim=action_dim,
            hidden_dims=advantage_dims,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling aggregation.
        
        Q(s, a) = V(s) + A(s, a) - mean(A(s, .))
        """
        features = self.shared(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine with mean-centered advantage
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values

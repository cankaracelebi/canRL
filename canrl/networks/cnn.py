"""Convolutional neural networks for visual observations."""

from typing import Sequence
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """
    Nature DQN-style CNN for visual observations.
    
    Architecture from "Human-level control through deep RL" (Mnih et al., 2015).
    Expects input shape (batch, channels, 84, 84).
    
    Args:
        input_channels: Number of input channels (e.g., 4 for frame stack).
        output_dim: Output feature dimension.
        
    Example:
        >>> # For Atari with 4 stacked grayscale frames
        >>> cnn = NatureCNN(input_channels=4, output_dim=512)
        >>> features = cnn(torch.randn(32, 4, 84, 84))  # (32, 512)
    """
    
    def __init__(self, input_channels: int = 4, output_dim: int = 512):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output size after conv layers
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, 84, 84)
            conv_out_size = self.conv(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.ReLU(),
        )
        
        self.output_dim = output_dim
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize with orthogonal weights."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects input in [0, 255] or [0, 1] range."""
        # Normalize if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        return self.fc(self.conv(x))


class ImpalaCNN(nn.Module):
    """
    IMPALA-style CNN with residual blocks.
    
    More powerful architecture for complex visual environments.
    Reference: "IMPALA: Scalable Distributed Deep-RL" (Espeholt et al., 2018)
    
    Args:
        input_channels: Number of input channels.
        output_dim: Output feature dimension.
        channels: Number of channels for each residual stack.
        
    Example:
        >>> cnn = ImpalaCNN(input_channels=4, output_dim=256)
        >>> features = cnn(torch.randn(32, 4, 84, 84))
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 256,
        channels: Sequence[int] = (16, 32, 32),
    ):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            layers.append(self._make_residual_stack(in_channels, out_channels))
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
        
        # Compute flattened size
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, 84, 84)
            conv_out_size = self.conv(sample).view(1, -1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(conv_out_size, output_dim),
            nn.ReLU(),
        )
        
        self.output_dim = output_dim
    
    def _make_residual_stack(
        self, in_channels: int, out_channels: int
    ) -> nn.Module:
        """Create a residual stack."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _ResidualBlock(out_channels),
            _ResidualBlock(out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.max() > 1.0:
            x = x / 255.0
        
        return self.fc(self.conv(x))


class _ResidualBlock(nn.Module):
    """Residual block for IMPALA CNN."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

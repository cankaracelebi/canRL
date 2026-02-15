"""Configuration management for experiments."""

from typing import Any
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """
    Configuration container with YAML/JSON support.
    
    Store experiment hyperparameters and settings with
    easy serialization.
    
    Example:
        >>> config = Config(
        ...     env_name="CartPole-v1",
        ...     learning_rate=0.001,
        ...     gamma=0.99,
        ...     custom={"hidden_dims": [64, 64]},
        ... )
        >>> config.save("config.yaml")
        >>> 
        >>> # Load later
        >>> config = Config.load("config.yaml")
    """
    
    # Environment
    env_name: str = "CartPole-v1"
    
    # Training
    total_steps: int = 100_000
    warmup_steps: int = 1000
    batch_size: int = 32
    update_frequency: int = 1
    
    # Agent
    learning_rate: float = 1e-3
    gamma: float = 0.99
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10_000
    
    # Buffer
    buffer_size: int = 10_000
    
    # Evaluation
    eval_frequency: int = 5_000
    eval_episodes: int = 10
    
    # Logging
    log_frequency: int = 1000
    log_dir: str = "runs"
    
    # Misc
    seed: int = 42
    device: str = "auto"
    
    # Custom fields
    custom: dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str | Path) -> None:
        """Save config to YAML or JSON."""
        path = Path(path)
        data = asdict(self)
        
        if path.suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load config from YAML or JSON."""
        path = Path(path)
        
        if path.suffix in (".yaml", ".yml"):
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def update(self, **kwargs: Any) -> "Config":
        """Return new config with updated values."""
        data = self.to_dict()
        data.update(kwargs)
        return Config.from_dict(data)
    
    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

"""Reproducibility utilities for seeding random number generators."""

import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    
    Seeds Python, NumPy, and PyTorch random number generators.
    
    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic CUDA operations.
                      May reduce performance.
    
    Example:
        >>> set_seed(42)
        >>> # Now training should be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # PyTorch 1.8+
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def seed_env(env: Any, seed: int) -> None:
    """
    Seed a Gymnasium environment.
    
    Args:
        env: Gymnasium environment.
        seed: Random seed.
    """
    env.reset(seed=seed)
    
    if hasattr(env, "action_space"):
        env.action_space.seed(seed)
    
    if hasattr(env, "observation_space"):
        env.observation_space.seed(seed)

"""Checkpoint utilities for saving and loading models."""

from typing import Any
from pathlib import Path
import json

import torch


class Checkpoint:
    """
    Save and load model checkpoints with metadata.
    
    Handles PyTorch state dicts, optimizer states, and
    arbitrary metadata for reproducibility.
    
    Example:
        >>> # Saving
        >>> checkpoint = Checkpoint()
        >>> checkpoint.save(
        ...     path="checkpoints/step_10000.pt",
        ...     model=q_network,
        ...     optimizer=optimizer,
        ...     step=10000,
        ...     config={"lr": 0.001},
        ... )
        >>> 
        >>> # Loading
        >>> data = checkpoint.load("checkpoints/step_10000.pt")
        >>> q_network.load_state_dict(data["model"])
        >>> optimizer.load_state_dict(data["optimizer"])
    """
    
    @staticmethod
    def save(
        path: str | Path,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        **extra_data: Any,
    ) -> None:
        """
        Save checkpoint to disk.
        
        Args:
            path: File path to save to.
            model: PyTorch model to save.
            optimizer: Optimizer to save.
            step: Current training step.
            **extra_data: Additional data to save (must be serializable).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": step,
            **extra_data,
        }
        
        if model is not None:
            checkpoint["model"] = model.state_dict()
        
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        
        # Also save readable metadata
        meta_path = path.with_suffix(".json")
        meta = {
            "step": step,
            **{k: v for k, v in extra_data.items() if _is_json_serializable(v)},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    @staticmethod
    def load(
        path: str | Path,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """
        Load checkpoint from disk.
        
        Args:
            path: File path to load from.
            model: If provided, load weights directly into model.
            optimizer: If provided, load state directly into optimizer.
            device: Device to load tensors to.
            
        Returns:
            Dictionary containing checkpoint data.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        
        if model is not None and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        return checkpoint
    
    @staticmethod
    def get_latest(checkpoint_dir: str | Path, pattern: str = "checkpoint_*.pt") -> Path | None:
        """
        Find the latest checkpoint in a directory.
        
        Args:
            checkpoint_dir: Directory to search.
            pattern: Glob pattern for checkpoint files.
            
        Returns:
            Path to latest checkpoint, or None if not found.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        return max(checkpoints, key=lambda p: p.stat().st_mtime)


def _is_json_serializable(obj: Any) -> bool:
    """Check if object can be JSON serialized."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False

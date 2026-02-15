"""Logging utilities for TensorBoard and console output."""

from typing import Any
from pathlib import Path
import json
import time


class Logger:
    """
    Unified logging to TensorBoard and console.
    
    Args:
        log_dir: Directory for TensorBoard logs and metrics.
        use_tensorboard: Whether to log to TensorBoard.
        
    Example:
        >>> logger = Logger("runs/experiment_1")
        >>> logger.log_scalar("train/loss", 0.5, step=100)
        >>> logger.log_dict({"lr": 0.001, "eps": 0.1}, step=100)
    """
    
    def __init__(self, log_dir: str | Path, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self._writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
        
        # Also log to JSON for easy parsing
        self._metrics_file = self.log_dir / "metrics.jsonl"
        self._start_time = time.time()
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._writer:
            self._writer.add_scalar(tag, value, step)
        
        self._write_json({
            "type": "scalar",
            "tag": tag,
            "value": value,
            "step": step,
            "time": time.time() - self._start_time,
        })
    
    def log_dict(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """Log multiple scalars from a dictionary."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram of values."""
        if self._writer:
            self._writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        if self._writer:
            self._writer.add_text(tag, text, step)
        
        self._write_json({
            "type": "text",
            "tag": tag,
            "value": text,
            "step": step,
        })
    
    def log_hyperparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        """Log hyperparameters."""
        if self._writer:
            self._writer.add_hparams(hparams, metrics or {})
        
        # Save to separate file for easy access
        hparams_file = self.log_dir / "hparams.json"
        with open(hparams_file, "w") as f:
            json.dump(hparams, f, indent=2)
    
    def _write_json(self, data: dict) -> None:
        """Append to JSONL metrics file."""
        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    
    def flush(self) -> None:
        """Flush all pending writes."""
        if self._writer:
            self._writer.flush()
    
    def close(self) -> None:
        """Close the logger."""
        if self._writer:
            self._writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

"""Training callbacks for extensible training hooks."""

from abc import ABC
from typing import Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from canrl.training.trainer import Trainer


class Callback(ABC):
    """
    Base callback class for training hooks.
    
    Inherit and override methods to add custom behavior
    during training.
    
    Example:
        class PrintProgressCallback(Callback):
            def on_episode_end(self, trainer, episode_return, episode_length):
                print(f"Episode {trainer.episodes}: return={episode_return:.2f}")
    """
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass
    
    def on_step(self, trainer: "Trainer") -> None:
        """Called after each environment step."""
        pass
    
    def on_episode_end(
        self, trainer: "Trainer", episode_return: float, episode_length: int
    ) -> None:
        """Called at the end of each episode."""
        pass
    
    def on_update(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        """Called after each agent update."""
        pass
    
    def on_eval(self, trainer: "Trainer", eval_stats: dict[str, float]) -> None:
        """Called after each evaluation."""
        pass


class CallbackList(Callback):
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)
    
    def on_train_end(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)
    
    def on_step(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            cb.on_step(trainer)
    
    def on_episode_end(
        self, trainer: "Trainer", episode_return: float, episode_length: int
    ) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(trainer, episode_return, episode_length)
    
    def on_update(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_update(trainer, metrics)
    
    def on_eval(self, trainer: "Trainer", eval_stats: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_eval(trainer, eval_stats)


class CheckpointCallback(Callback):
    """
    Save agent checkpoints during training.
    
    Args:
        save_dir: Directory to save checkpoints.
        save_frequency: Steps between saves.
        save_best: Whether to save best model based on eval.
        
    Example:
        >>> callback = CheckpointCallback(
        ...     save_dir="checkpoints",
        ...     save_frequency=10000,
        ...     save_best=True,
        ... )
    """
    
    def __init__(
        self,
        save_dir: str | Path,
        save_frequency: int = 10000,
        save_best: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.save_best = save_best
        
        self.best_return = float("-inf")
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_step(self, trainer: "Trainer") -> None:
        if trainer.total_steps % self.save_frequency == 0:
            path = self.save_dir / f"checkpoint_{trainer.total_steps}.pt"
            trainer.agent.save(path)
    
    def on_eval(self, trainer: "Trainer", eval_stats: dict[str, float]) -> None:
        if self.save_best:
            mean_return = eval_stats.get("mean_return", float("-inf"))
            if mean_return > self.best_return:
                self.best_return = mean_return
                path = self.save_dir / "best_model.pt"
                trainer.agent.save(path)


class EarlyStoppingCallback(Callback):
    """
    Stop training when target performance is reached.
    
    Args:
        target_return: Stop when eval mean return exceeds this.
        patience: Number of evals without improvement before stopping.
    """
    
    def __init__(
        self,
        target_return: float | None = None,
        patience: int | None = None,
    ):
        self.target_return = target_return
        self.patience = patience
        
        self.best_return = float("-inf")
        self.evals_without_improvement = 0
        self.should_stop = False
    
    def on_eval(self, trainer: "Trainer", eval_stats: dict[str, float]) -> None:
        mean_return = eval_stats.get("mean_return", float("-inf"))
        
        # Check target
        if self.target_return is not None and mean_return >= self.target_return:
            print(f"Target return {self.target_return} reached! Stopping.")
            self.should_stop = True
        
        # Check patience
        if self.patience is not None:
            if mean_return > self.best_return:
                self.best_return = mean_return
                self.evals_without_improvement = 0
            else:
                self.evals_without_improvement += 1
                
            if self.evals_without_improvement >= self.patience:
                print(f"No improvement for {self.patience} evals. Stopping.")
                self.should_stop = True


class LoggingCallback(Callback):
    """
    Print training progress to console.
    
    Args:
        log_frequency: Episodes between log messages.
        
    Example:
        >>> callback = LoggingCallback(log_frequency=10)
        # Prints every 10 episodes
    """
    
    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency
        self.recent_returns: list[float] = []
    
    def on_episode_end(
        self, trainer: "Trainer", episode_return: float, episode_length: int
    ) -> None:
        self.recent_returns.append(episode_return)
        
        if trainer.episodes % self.log_frequency == 0:
            import numpy as np
            mean_return = np.mean(self.recent_returns[-self.log_frequency:])
            
            print(
                f"Episode {trainer.episodes:5d} | "
                f"Step {trainer.total_steps:7d} | "
                f"Return: {episode_return:7.2f} | "
                f"Mean({self.log_frequency}): {mean_return:7.2f}"
            )
    
    def on_eval(self, trainer: "Trainer", eval_stats: dict[str, float]) -> None:
        print(
            f"[EVAL] Step {trainer.total_steps:7d} | "
            f"Mean: {eval_stats['mean_return']:7.2f} ± {eval_stats['std_return']:.2f}"
        )

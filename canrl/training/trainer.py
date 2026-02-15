"""Main training loop for RL agents."""

from typing import Any
import time
import gymnasium as gym
import numpy as np

from canrl.agents.base_agent import BaseAgent
from canrl.buffers.base_buffer import BaseBuffer, Transition
from canrl.training.callbacks import Callback, CallbackList
from canrl.training.evaluator import Evaluator
from canrl.utils.logger import Logger


class Trainer:
    """
    Main training loop for RL agents.
    
    Handles environment interaction, experience collection,
    agent updates, and logging.
    
    Args:
        agent: The RL agent to train.
        env: Training environment.
        buffer: Experience replay buffer.
        logger: Logger for metrics.
        evaluator: Optional evaluator for periodic evaluation.
        callbacks: List of training callbacks.
        
    Example:
        >>> trainer = Trainer(
        ...     agent=my_dqn_agent,
        ...     env=gym.make("CartPole-v1"),
        ...     buffer=ReplayBuffer(10000, state_shape=(4,)),
        ...     logger=Logger("runs/cartpole"),
        ... )
        >>> trainer.train(total_steps=100000)
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env,
        buffer: BaseBuffer,
        logger: Logger | None = None,
        evaluator: Evaluator | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.logger = logger
        self.evaluator = evaluator
        self.callbacks = CallbackList(callbacks or [])
        
        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.episode_return = 0.0
        self.episode_length = 0
        
        self._state: np.ndarray | None = None
        self._start_time = 0.0
    
    def train(
        self,
        total_steps: int,
        warmup_steps: int = 1000,
        update_frequency: int = 1,
        batch_size: int = 32,
        eval_frequency: int = 10000,
        log_frequency: int = 1000,
    ) -> dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            total_steps: Total environment steps to train for.
            warmup_steps: Steps before starting updates (fill buffer).
            update_frequency: Steps between agent updates.
            batch_size: Batch size for updates.
            eval_frequency: Steps between evaluations (0 to disable).
            log_frequency: Steps between logging.
            
        Returns:
            Dictionary of final training statistics.
        """
        self._start_time = time.time()
        self._reset_episode()
        
        self.callbacks.on_train_begin(self)
        
        while self.total_steps < total_steps:
            # Collect experience
            action = self.agent.select_action(self._state, deterministic=False)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            transition = Transition(
                state=self._state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,  # Use terminated, not truncated for bootstrap
            )
            self.buffer.add(transition)
            
            self._state = next_state
            self.episode_return += reward
            self.episode_length += 1
            self.total_steps += 1
            
            # Update agent
            if (
                self.total_steps >= warmup_steps
                and self.total_steps % update_frequency == 0
                and self.buffer.can_sample(batch_size)
            ):
                batch = self.buffer.sample(batch_size)
                metrics = self.agent.update(batch)
                
                self.callbacks.on_update(self, metrics)
                
                if self.logger and self.total_steps % log_frequency == 0:
                    for key, value in metrics.items():
                        self.logger.log_scalar(f"train/{key}", value, self.total_steps)
            
            # Handle episode end
            if done:
                self._on_episode_end(info)
                self._reset_episode()
            
            # Periodic evaluation
            if (
                eval_frequency > 0
                and self.evaluator is not None
                and self.total_steps % eval_frequency == 0
            ):
                self._evaluate()
            
            self.callbacks.on_step(self)
        
        self.callbacks.on_train_end(self)
        
        return self._get_final_stats()
    
    def _reset_episode(self) -> None:
        """Reset for new episode."""
        self._state, _ = self.env.reset()
        self.episode_return = 0.0
        self.episode_length = 0
    
    def _on_episode_end(self, info: dict) -> None:
        """Handle episode completion."""
        self.episodes += 1
        
        if self.logger:
            self.logger.log_scalar("rollout/episode_return", self.episode_return, self.total_steps)
            self.logger.log_scalar("rollout/episode_length", self.episode_length, self.total_steps)
            self.logger.log_scalar("rollout/episodes", self.episodes, self.total_steps)
        
        self.callbacks.on_episode_end(self, self.episode_return, self.episode_length)
    
    def _evaluate(self) -> None:
        """Run evaluation."""
        self.agent.eval_mode()
        eval_stats = self.evaluator.evaluate(self.agent)
        self.agent.train_mode()
        
        if self.logger:
            for key, value in eval_stats.items():
                self.logger.log_scalar(f"eval/{key}", value, self.total_steps)
        
        self.callbacks.on_eval(self, eval_stats)
    
    def _get_final_stats(self) -> dict[str, Any]:
        """Compile final training statistics."""
        elapsed = time.time() - self._start_time
        
        return {
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "elapsed_time": elapsed,
            "steps_per_second": self.total_steps / elapsed if elapsed > 0 else 0,
        }

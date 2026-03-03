"""
canRL - A modular Reinforcement Learning framework.

This framework provides infrastructure for RL research:
- Environment wrappers
- Replay buffers
- Neural network building blocks
- Training utilities

Implement your own algorithms in canrl/algorithms/
"""

__version__ = "0.1.0"

from canrl.envs import (
    FrameStack,
    NormalizeObservation,
    NormalizeReward,
    ActionRepeat,
    Monitor,
)
from canrl.buffers import ReplayBuffer, PrioritizedReplayBuffer
from canrl.algorithms.base import BaseAgent
from canrl.training import Trainer, Evaluator
from canrl.utils import Logger, Checkpoint, Config, set_seed

__all__ = [
    # Env wrappers
    "FrameStack",
    "NormalizeObservation",
    "NormalizeReward",
    "ActionRepeat",
    "Monitor",
    # Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    # Agents
    "BaseAgent",
    # Training
    "Trainer",
    "Evaluator",
    # Utils
    "Logger",
    "Checkpoint",
    "Config",
    "set_seed",
]

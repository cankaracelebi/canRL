"""
Algorithm implementations.

Organized by category:
- value_based/   : Q-Learning, DQN, DDQN, etc.
- policy_gradient/: PPO, A2C, SAC, TD3, etc.
"""

from canrl.algorithms.base import BaseAgent
from canrl.algorithms.value_based.dqn import DQN
from canrl.algorithms.value_based.q_learning import QLearningAgent

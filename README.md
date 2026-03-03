# canRL 

>  **Work in Progress** - This framework is under active development. Core infrastructure is functional, but algorithms are still being implemented.

A robust Reinforcement Learning framework for building and experimenting with RL algorithms. canRL aims to contain a wide variety of RL algorithms from literature ready to train!

## Vision

canRL aims to provide a complete, production-ready RL framework with:
- **Modular Components** - Environment wrappers, replay buffers, neural networks
- **Training Infrastructure** - Trainer, evaluator, callbacks, and logging
- **Algorithm Library** - Common RL algorithms (DQN, PPO, SAC, etc.) *(in progress)*
- **Experiment Tools** - TensorBoard logging, checkpointing, configuration management
- **Extensible Design** - Easy to customize and add new components

## Current Status

**Completed:**
-  Environment wrappers (frame stack, normalization, action repeat, monitoring)
-  Replay buffers (uniform and prioritized)
-  Neural network components (MLP, CNN architectures)
-  Training infrastructure (trainer, evaluator, callbacks)
-  Utilities (logging, checkpointing, scheduling, seeding)

**In Development:**
-  Algorithm implementations (templates and skeletons provided)
-  Documentation and examples
-  Unit tests

## Installation

```bash
git clone https://github.com/yourusername/canRL.git
cd canRL
pip install -e .
```

**Requirements:** Python ≥3.10, PyTorch ≥2.0.0

## Project Structure

```
canrl/
├── envs/           # Environment wrappers
│   ├── frame_stack.py      # Stack frames for temporal info
│   ├── normalize.py        # Observation/reward normalization
│   ├── action_repeat.py    # Frame skipping
│   └── monitor.py          # Episode tracking
├── buffers/        # Experience replay
│   ├── replay_buffer.py    # Uniform sampling
│   └── prioritized.py      # Prioritized experience replay
├── networks/       # Neural network building blocks
│   ├── mlp.py              # MLP and Dueling networks
│   └── cnn.py              # Nature CNN, IMPALA CNN
├── training/       # Training infrastructure
│   ├── trainer.py          # Main training loop
│   ├── evaluator.py        # Evaluation utilities
│   └── callbacks.py        # Training callbacks
├── utils/          # Utilities
│   ├── logger.py           # TensorBoard logging
│   ├── checkpoint.py       # Model saving/loading
│   ├── config.py           # Configuration management
│   ├── schedule.py         # Epsilon/LR schedules
│   └── seed.py             # Reproducibility
└── algorithms/     # Algorithm implementations
    ├── base.py             # BaseAgent abstract class
    ├── value_based/        # Value-based methods
    │   ├── q_learning.py   # Tabular Q-Learning
    │   ├── dqn.py          # Deep Q-Network
    │   ├── ddqn.py         # Double DQN
    │   └── masked_dqn.py   # Masked DQN
    └── policy_gradient/    # Policy gradient & actor-critic
        └── (PPO, A2C, SAC, TD3 — planned)
```

## Quick Start

### 1. Run Example (Framework Demo)

```bash
python examples/train_example.py
```


### 2. Train Your Agent



## Components Reference

### Environment Wrappers

```python
from canrl import FrameStack, NormalizeObservation, ActionRepeat, Monitor

# Stack 4 frames for Atari-style environments
env = FrameStack(env, num_stack=4)

# Normalize observations to zero mean, unit variance
env = NormalizeObservation(env)

# Repeat each action 4 times (frame skipping)
env = ActionRepeat(env, repeat=4)

# Track episode statistics
env = Monitor(env)
print(env.episode_returns)  # List of episode returns
```

### Replay Buffers

```python
from canrl import ReplayBuffer, PrioritizedReplayBuffer
from canrl.buffers import Transition

# Standard buffer
buffer = ReplayBuffer(capacity=100000, state_shape=(4,))
buffer.add(Transition(state, action, reward, next_state, done))
batch = buffer.sample(batch_size=32)

# Prioritized replay
buffer = PrioritizedReplayBuffer(capacity=100000, state_shape=(4,), alpha=0.6)
batch = buffer.sample(32)
buffer.update_priorities(batch.indices, td_errors)  # Update with TD errors
```

### Neural Networks

```python
from canrl.networks import MLP, DuelingMLP, NatureCNN

# Standard MLP
q_net = MLP(input_dim=4, output_dim=2, hidden_dims=(256, 256))

# Dueling architecture
q_net = DuelingMLP(input_dim=4, action_dim=2)

# CNN for images (84x84)
encoder = NatureCNN(input_channels=4, output_dim=512)
```

### Training

```python
from canrl import Trainer, Evaluator
from canrl.training import CheckpointCallback, LoggingCallback

# Evaluator for periodic evaluation
evaluator = Evaluator(eval_env, num_episodes=10)

# Callbacks
callbacks = [
    CheckpointCallback("checkpoints/", save_frequency=10000),
    LoggingCallback(log_frequency=10),
]

# Trainer with all components
trainer = Trainer(
    agent=agent,
    env=env,
    buffer=buffer,
    logger=logger,
    evaluator=evaluator,
    callbacks=callbacks,
)

trainer.train(
    total_steps=100000,
    warmup_steps=1000,
    batch_size=32,
    eval_frequency=5000,
)
```

### Utilities

```python
from canrl import Logger, Checkpoint, Config, set_seed
from canrl.utils import LinearSchedule, ExponentialSchedule

# Seeding
set_seed(42, deterministic=True)

# Logging
logger = Logger("runs/experiment")
logger.log_scalar("train/loss", 0.5, step=100)

# Checkpointing
Checkpoint.save("model.pt", model=network, optimizer=opt, step=1000)
data = Checkpoint.load("model.pt", model=network)

# Config
config = Config(env_name="CartPole-v1", learning_rate=1e-3)
config.save("config.yaml")

# Schedules
epsilon = LinearSchedule(start=1.0, end=0.01, duration=10000)
print(epsilon(5000))  # 0.505
```

## Roadmap

- [ ] Complete value-based algorithms (DQN, DDQN, Dueling DQN, Rainbow)
- [ ] Implement policy gradient algorithms (REINFORCE, A2C, PPO)
- [ ] Implement actor-critic algorithms (SAC, TD3, DDPG)
- [ ] Add agent performance tests
- [ ] Benchmarking suite (CartPole, LunarLander, Atari)
- [ ] Multi-GPU / distributed training support

## License

MIT License

## Acknowledgments

This framework is under active development. 

"""
Example training script demonstrating canRL usage.

This example uses the TemplateAgent to verify that
the framework components work together.
"""

import gymnasium as gym
import numpy as np

from canrl import (
    ReplayBuffer,
    Monitor,
    Trainer,
    Evaluator,
    Logger,
    Config,
    set_seed,
)
from canrl.buffers import Transition
from canrl.training import LoggingCallback
from canrl.algorithms import DQN

def main():
    """Run example training."""
    # Configuration
    config = Config(
        env_name="CartPole-v1",
        total_steps=50000,
        warmup_steps=500,
        batch_size=32,
        seed=42,
        buffer_size=10000,
    
        
    )
    
    print("=" * 50)
    print("canRL Framework Example")
    print("=" * 50)
    print(f"\nConfig: {config.env_name}")
    print(f"Total steps: {config.total_steps}")
    print(f"Device: {config.get_device()}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create environment with monitoring
    env = Monitor(gym.make(config.env_name))
    eval_env = gym.make(config.env_name)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\nState dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    
    # Create a simple random agent for demonstration
    # (Replace with your implemented agent!)
    class RandomAgent:
        """Simple random agent for testing framework."""
        
        def __init__(self, action_dim):
            self.action_dim = action_dim
        
        def select_action(self, state, deterministic=False):
            return np.random.randint(self.action_dim)
        
        def update(self, batch):
            # Pretend to do something
            return {"fake_loss": np.random.random()}
        
        def save(self, path):
            pass
        
        def load(self, path):
            pass
        
        def train_mode(self):
            pass
        
        def eval_mode(self):
            pass
    
    # agent = RandomAgent(action_dim)
    agent = DQN(state_dim, action_dim)
    
    
    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=config.buffer_size,
        state_shape=(state_dim,),
    )
    
    # Create logger
    logger = Logger(f"{config.log_dir}/example_run", use_tensorboard=False)
    
    # Create evaluator
    evaluator = Evaluator(eval_env, num_episodes=3)
    
    # Create trainer with callbacks
    callbacks = [LoggingCallback(log_frequency=5)]
    
    trainer = Trainer(
        agent=agent,
        env=env,
        buffer=buffer,
        logger=logger,
        evaluator=evaluator,
        callbacks=callbacks,
    )
    
    print("\n" + "-" * 50)
    print("Starting training...")
    print("-" * 50 + "\n")
    
    # Run training
    stats = trainer.train(
        total_steps=config.total_steps,
        warmup_steps=config.warmup_steps,
        batch_size=config.batch_size,
        eval_frequency=1000,
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nFinal Statistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Episodes: {stats['episodes']}")
    print(f"  Time: {stats['elapsed_time']:.2f}s")
    print(f"  Speed: {stats['steps_per_second']:.1f} steps/s")
    
    # Show monitor stats
    monitor_stats = env.get_statistics()
    if monitor_stats:
        print(f"\nEpisode Statistics:")
        print(f"  Mean return: {monitor_stats['mean_return']:.2f}")
        print(f"  Max return: {monitor_stats['max_return']:.2f}")
    
    print("\n Framework test passed!")
    
    # Save config for reference
    config.save("runs/example_run/config.yaml")

    logger.close()
    env.close()
    eval_env.close()

    # Render trained agent
    print("\n" + "-" * 50)
    print("Rendering trained agent...")
    print("-" * 50 + "\n")

    render_env = gym.make(config.env_name, render_mode="human")
    render_evaluator = Evaluator(render_env, num_episodes=3, render=True)
    render_stats = render_evaluator.evaluate(agent)
    print(f"Render eval mean return: {render_stats['mean_return']:.2f}")
    render_env.close()


if __name__ == "__main__":
    main()

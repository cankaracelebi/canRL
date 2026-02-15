"""
Basic tests for canRL components.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def test_buffer_add_and_sample(self):
        """Test adding transitions and sampling."""
        from canrl.buffers import ReplayBuffer
        from canrl.buffers.base_buffer import Transition
        
        buffer = ReplayBuffer(capacity=100, state_shape=(4,))
        
        # Add some transitions
        for i in range(50):
            transition = Transition(
                state=np.random.randn(4).astype(np.float32),
                action=np.random.randint(0, 2),
                reward=np.random.randn(),
                next_state=np.random.randn(4).astype(np.float32),
                done=False,
            )
            buffer.add(transition)
        
        assert len(buffer) == 50
        
        # Sample
        batch = buffer.sample(16)
        assert batch.states.shape == (16, 4)
        assert batch.actions.shape == (16,)
        assert batch.rewards.shape == (16,)
    
    def test_buffer_overflow(self):
        """Test circular buffer behavior."""
        from canrl.buffers import ReplayBuffer
        from canrl.buffers.base_buffer import Transition
        
        buffer = ReplayBuffer(capacity=10, state_shape=(2,))
        
        # Add more than capacity
        for i in range(25):
            buffer.add(Transition(
                state=np.array([i, i], dtype=np.float32),
                action=0,
                reward=0.0,
                next_state=np.array([i+1, i+1], dtype=np.float32),
                done=False,
            ))
        
        # Should stay at capacity
        assert len(buffer) == 10


class TestEnvironmentWrappers:
    """Test environment wrappers."""
    
    def test_frame_stack(self):
        """Test frame stacking wrapper."""
        import gymnasium as gym
        from canrl.envs import FrameStack
        
        env = gym.make("CartPole-v1")
        env = FrameStack(env, num_stack=4)
        
        obs, _ = env.reset()
        assert obs.shape[0] == 4  # 4 stacked frames
        
        obs, *_ = env.step(0)
        assert obs.shape[0] == 4
        
        env.close()
    
    def test_monitor(self):
        """Test episode monitoring."""
        import gymnasium as gym
        from canrl.envs import Monitor
        
        env = Monitor(gym.make("CartPole-v1"))
        
        obs, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        
        # Should have recorded one episode
        assert env.total_episodes >= 1
        
        env.close()


class TestNetworks:
    """Test neural network modules."""
    
    def test_mlp(self):
        """Test MLP forward pass."""
        import torch
        from canrl.networks import MLP
        
        net = MLP(input_dim=4, output_dim=2, hidden_dims=(32, 32))
        x = torch.randn(8, 4)
        y = net(x)
        
        assert y.shape == (8, 2)
    
    def test_nature_cnn(self):
        """Test CNN forward pass."""
        import torch
        from canrl.networks import NatureCNN
        
        cnn = NatureCNN(input_channels=4, output_dim=256)
        x = torch.randn(2, 4, 84, 84)
        y = cnn(x)
        
        assert y.shape == (2, 256)


class TestSchedules:
    """Test learning rate and epsilon schedules."""
    
    def test_linear_schedule(self):
        """Test linear interpolation."""
        from canrl.utils import LinearSchedule
        
        schedule = LinearSchedule(start=1.0, end=0.1, duration=1000)
        
        assert schedule(0) == 1.0
        assert abs(schedule(500) - 0.55) < 0.01
        assert schedule(1000) == 0.1
        assert schedule(2000) == 0.1  # Should stay at end


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test config creation and access."""
        from canrl.utils import Config
        
        config = Config(
            env_name="CartPole-v1",
            learning_rate=0.001,
            gamma=0.99,
        )
        
        assert config.env_name == "CartPole-v1"
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99
    
    def test_config_save_load(self, tmp_path):
        """Test config serialization."""
        from canrl.utils import Config
        
        config = Config(env_name="TestEnv", seed=123)
        path = tmp_path / "config.yaml"
        config.save(path)
        
        loaded = Config.load(path)
        assert loaded.env_name == "TestEnv"
        assert loaded.seed == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

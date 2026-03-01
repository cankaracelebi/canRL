"""
Tabular Q-Learning on Taxi-v3.

Taxi-v3 is a fully discrete environment (500 states, 6 actions),
making it a natural fit for tabular Q-learning with no discretization needed.

Actions: south(0), north(1), east(2), west(3), pickup(4), dropoff(5)
"""

import gymnasium as gym
import numpy as np

from canrl.algorithms import QLearningAgent
from canrl.utils.seed import set_seed


def main():
    num_episodes = 5000
    max_steps_per_episode = 200
    seed = 42

    set_seed(seed)

    env = gym.make("Taxi-v3")
    state_dim = env.observation_space.n  # 500
    action_dim = env.action_space.n      # 6

    agent = QLearningAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )

    print("=" * 50)
    print("Q-Learning on Taxi-v3")
    print("=" * 50)
    print(f"  States: {state_dim}, Actions: {action_dim}")
    print(f"  Episodes: {num_episodes}")
    print()

    # Training loop
    returns = []
    best_mean = -float("inf")

    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0.0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_step(state, action, reward, next_state, done)

            state = next_state
            episode_return += reward

            if done:
                break

        returns.append(episode_return)

        if (ep + 1) % 100 == 0:
            mean_ret = np.mean(returns[-100:])
            best_mean = max(best_mean, mean_ret)
            print(
                f"Episode {ep + 1:5d} | "
                f"Mean Return (100): {mean_ret:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Best: {best_mean:.1f}"
            )

    env.close()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"  Final mean (100): {np.mean(returns[-100:]):.1f}")
    print(f"  Best mean (100):  {best_mean:.1f}")

    # Render trained agent
    print("\n" + "-" * 50)
    print("Rendering trained agent...")
    print("-" * 50 + "\n")

    render_env = gym.make("Taxi-v3", render_mode="human")

    for ep in range(3):
        state, _ = render_env.reset()
        episode_return = 0.0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = render_env.step(action)
            episode_return += reward

            if terminated or truncated:
                break

        print(f"  Render episode {ep + 1}: return = {episode_return:.0f}")

    render_env.close()


if __name__ == "__main__":
    main()

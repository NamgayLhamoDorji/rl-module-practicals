import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Create environment (same as practical1)
env = gym.make("FrozenLake-v1", render_mode="null")

# Initialize Q-table with zeros
# Shape: (num_states, num_actions) = (16, 4)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

print(f"Q-table shape: {q_table.shape}")
print(f"Initial Q-table (first 5 states):")
print(q_table[:5])

def choose_action(state, q_table, epsilon):
    """
    Choose action using epsilon-greedy strategy

    Args:
        state: Current state number
        q_table: Current Q-table
        epsilon: Exploration rate (0 = always exploit, 1 = always explore)

    Returns:
        action: Chosen action number
    """
    if random.random() < epsilon:
        # EXPLORE: Choose random action
        return env.action_space.sample()
    else:
        # EXPLOIT: Choose action with highest Q-value
        return np.argmax(q_table[state])

# Q-learning hyperparameters
learning_rate = 0.1    # α - how much to update Q-values each step
discount_factor = 0.99 # γ - how much we value future rewards
epsilon = 1.0          # ε - exploration rate (start exploring everything)
epsilon_decay = 0.995  # Gradually reduce exploration
min_epsilon = 0.01     # Always explore at least 1%

# Training parameters
num_episodes = 10000
rewards_per_episode = []

print("Starting Q-learning training...")

for episode in range(num_episodes):
    # Reset environment for new episode
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Choose action using epsilon-greedy
        action = choose_action(state, q_table, epsilon)

        # Take action and observe result
        next_state, reward, terminated, truncated, info = env.step(action)

        # 🔑 Q-LEARNING UPDATE RULE - Step by Step
        # Current Q-value (our old belief about this action)
        current_q = q_table[state, action]

        # Best future Q-value (if not terminal)
        if terminated:
            max_future_q = 0  # No future rewards if episode ended
        else:
            max_future_q = np.max(q_table[next_state])  # Best action from next state

        # Calculate the "target" Q-value (what Q should be based on this experience)
        target_q = reward + discount_factor * max_future_q

        # Calculate the "prediction error" (how wrong were we?)
        error = target_q - current_q

        # Update Q-value by moving towards the target (scaled by learning rate)
        new_q = current_q + learning_rate * error

        # Store the updated Q-value in our table
        q_table[state, action] = new_q

        # Move to next state
        state = next_state
        total_reward += reward

    # Store episode reward
    rewards_per_episode.append(total_reward)

    # Decay epsilon (explore less over time)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])  # Last 100 episodes
        print(f"Episode {episode + 1}/10000, Average Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}")

print("Training completed!")

# Plot learning curve
plt.figure(figsize=(12, 4))

# Plot 1: Rewards over time (with moving average)
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode, alpha=0.3, label='Episode Reward')

# Calculate moving average for smoother trend
window_size = 100
moving_avg = []
for i in range(len(rewards_per_episode)):
    start = max(0, i - window_size)
    moving_avg.append(np.mean(rewards_per_episode[start:i+1]))

plt.plot(moving_avg, label='Moving Average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-Learning Training Progress')
plt.legend()

# Plot 2: Final Q-table heatmap
plt.subplot(1, 2, 2)
plt.imshow(q_table, cmap='viridis', aspect='auto')
plt.xlabel('Actions (0:Left, 1:Down, 2:Right, 3:Up)')
plt.ylabel('States')
plt.title('Final Q-Table Values')
plt.colorbar()

plt.tight_layout()
plt.show()

# Testing the trained agent
def test_agent(q_table, num_episodes=1000, render=False):
    """Test the trained Q-learning agent"""
    env_test = gym.make("FrozenLake-v1", render_mode="null" if render else None)

    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Use pure exploitation (epsilon = 0)
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward

            if render and episode < 3:  # Only render first 3 episodes
                env_test.render()
                time.sleep(1)

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

    env_test.close()

    success_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)

    return success_rate, avg_reward

# Test your Q-learning agent
success_rate, avg_reward = test_agent(q_table, num_episodes=1000)

print(f"Q-Learning Agent Results (1000 episodes):")
print(f"Success Rate: {success_rate:.1%}")
print(f"Average Reward: {avg_reward:.4f}")
print(f"\nComparison to Random Agent (~6% success rate):")
print(f"Improvement: {success_rate/0.06:.1f}x better!")
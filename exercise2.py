import gymnasium as gym

env = gym.make("FrozenLake-v1")

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# --- THE MAIN LOOP ---
# Create an outer loop to run 1000 episodes.
num_episodes = 1000
rewards_per_episode = []

for episode in range(num_episodes):
    observation, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    while not terminated and not truncated:
        action = env.action_space.sample()
        print(f"Taking action: {action} (0:L, 1:D, 2:R, 3:U)")

        next_observation, reward, terminated, truncated, info = env.step(action)

    
        episode_reward += reward
        observation = next_observation

    # Append the final reward for this episode to the list
    rewards_per_episode.append(episode_reward)

total_reward =sum(rewards_per_episode)
print(f"\nEpisode finished! Total Reward: {total_reward}")
    
average_reward = sum(rewards_per_episode) / num_episodes 
print(f"Average reward over {num_episodes} episodes: {average_reward:.4f}")


env.close()
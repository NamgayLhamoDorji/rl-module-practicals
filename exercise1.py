import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human") #Modify your script to use gym.make("CartPole-v1", render_mode="human")

# Print the action_space and observation_space.
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")


observation, info = env.reset()


terminated = False
truncated = False
total_reward = 0.0


while not terminated and not truncated:
    
    env.render()

    action = env.action_space.sample()
    print(f"Taking action: {action} (0:L, 1:D, 2:R, 3:U)")

    next_observation, reward, terminated, truncated, info = env.step(action)

    
    total_reward += reward
    observation = next_observation

   
    time.sleep(1)

print(f"\nEpisode finished! Total Reward: {total_reward}")


env.close()
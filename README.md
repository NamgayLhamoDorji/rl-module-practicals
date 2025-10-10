Practical 1
1.  What type of space is the action space? How many actions are there?
 The action space is Discrete(2). This means there are 2 possible actions: 0 (push the cart left) and 1 (push the cart right).

2.  What type of space is the observation space? 
   The output is Box(4,). 
   This represents a continuous space with 4 numbers. 
   Based on the problem, what could these four numbers possibly represent?
   The four numbers could possibly represent:
   1. Cart Position         - How far the cart is from the center? 
   2. Cart Velocity         - How fast the cart is moving?
   3. Pole Angle            - The tilt of the pole from vertical.
   4. Pole Angular Velocity - How fast the pole is tilting.

3. What does the reward seem to represent in this environment?
Running the random agent shows that the reward is 1 for every time step the pole remains balanced. The episode ends when the cart moves too far from the center.So, the reward is basically the number steps you survive before failing.

We were given two exercise to complete, were in exercise 1 we had to create "CartPole-v1" environment. Here I copied the code from index.py and replaced the "FrozenLake-v1" environment with "CartPole-v1"environment. In exercise 2 we had to use the "FrozenLake-v1" script as a base, remove the render(), print(), and time.sleep() calls from the main loop to speed up execution and create an outer loop to run 1000 episodes.Then we had to store the total_reward from each episode in a list and after the outer loop finishes, we need to insert code to calculate and print the average reward over the 1000 episodes. 

While doing the exercises, at first it was difficult and hard to understand what was really happenning and how was the code running. With time as I kept on practicing for 2 to 3 times and did the exercises all over again  for the submission. I got to understand it better and found it easy.

Practical 2
Implementation Summary

In this practical, we implemented a Q-Learning agent to explore and learn an optimal policy for navigating a simple grid-world environment. The agent started with no prior knowledge and gradually improved by interacting with the environment, collecting rewards, and updating its Q-table using the Bellman equation.

An ε-greedy strategy was applied to balance exploration (trying new actions to discover rewards) and exploitation (choosing the best-known action to maximize gain). Training ran for 10,000 episodes, during which the agent refined its policy through repeated experience and feedback.

Main implementation steps included initializing the Q-table to zeros, performing iterative Q-value updates using the learning rate (α) and discount factor (γ), gradually reducing ε to shift from exploration to exploitation, and continuously monitoring performance through reward tracking and Q-table visualization.

Performance Results

After 10,000 training episodes the Q-Learning agent achieved a final success rate of 71.4%, with an average reward of 0.714. By contrast, a random agent achieved roughly 6% success. The agent displayed consistent improvement during training and reached a stable performance level by the end of the run.

Training Progress Analysis

The training progress plot (moving average of rewards) shows a steady upward trend. Early episodes exhibited large fluctuations due to random exploration, but over time the orange moving-average line smoothed and stabilized in the 0.6–0.7 range, indicating the agent learned an effective and repeatable strategy. This confirms that, given sufficient episodes, Q-learning can REACH toward a near-optimal policy.

Q-Table Analysis

The final Q-table heatmap highlights the learned preferences across states and actions. Brighter regions indicate higher Q-values (stronger preferences), while darker areas correspond to low-value or infrequently visited states. Most states show a clear best action, demonstrating that the agent successfully differentiated good from bad actions and learned where to move to maximize long-term reward.

Hyperparameter Observations

During experimentation the following hyperparameter settings produced stable learning:

Learning rate α ≈ 0.1 — stable incremental updates without overshooting.

Discount factor γ = 0.99 — strong emphasis on long-term rewards.

Exploration rate ε decayed gradually to 0.01 — allowed ample early exploration then promoted exploitation.

These hyperparameters strongly influenced convergence speed and stability and tuning them was critical to reach the reported success rate.

Challenges Faced

Key challenges included balancing exploration vs. exploitation (ε scheduling), handling early reward fluctuations (necessitating smoothing like moving averages) and tuning the learning rate (high α caused oscillations while low α slowed learning).

Key Insights

This exercise reinforced that even a simple tabular Q-learning algorithm can reach strong performance with proper hyperparameter tuning and patience. The moving-average reward curve made clear that meaningful improvement often appears only after thousands of episodes. ε-decay was crucial for the transition from exploration to stable exploitation, and the Q-table served as an interpretable representation of the agent’s learned knowledge.

Learning Reflection

Q-Learning proved to be a powerful, intuitive method for learning policies directly from interaction without a model of the environment. Compared to a random agent, the Q-learning agent learned structured decision-making that prioritized long-term reward, illustrating how reinforcement learning systems can autonomously improve through trial and error. 
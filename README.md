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
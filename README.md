# BipedalWalker-v3
Data-driven Project based on BipedalWalker-v3 of OpenAi

This repo is intended to describe the experience of project occurred during teaching of Control System Design of the course degree in Computer Engineering, at the University of Salerno.
The project activity involves the description of some reinforcement algorithms, existing in the literature learning and their application on the environment openAI called BipedalWalker. They were implemented some of the latest approaches of RL for robotic locomotion, initially they were considered simple algorithms and then move on to more complex and advanced algorithms.

## Goal
The main goal of this project is that to solve the problem characterized by the environment  BipedalWalker-v3. The problem arises as concluded when the robot chooses correct actions for
don't fall and walk to the end of the path.
In particular, the reward returned by the environment is a value of the continuous space which depends on the distance the robot travels. Formally the resolution of the problem occurs when the robot gets an average reward greater than 300 (path completed) for 100 consecutive episodes:

We therefore want to find a reinforcement algorithm learning able to achieve the goal mentioned in the shortest possible time.

## Algorithms Used
- Q-Learning
- DQN
- DDPG
- TD3

## Results
Unlike everyone the other algorithms the TD3 manages to achieve the goal, in fact in some episodes the DDPG manages to reach the maximum reward but can't reach convergence. This is attributable to the fact that DDPG has many limitations on the specific task, however an improvement on the TD3 was
able to successfully train people. 
The final success is due to those differences specifications between TD3 and DDPG, which made it possible to stabilize learning by reducing variance. Moreover, TD3 solves the problem of estimation errors that arise they accumulate over time and can carry the agent in excellent premises. The following table shows them the final results that summarize the experiments performed with all algorithms.

| Algorithm | Average max score | Max score | Episodes | Time in hours | 
|-----------|-------------------|-----------|----------|---------------|
|Q-Learning |-76                |-50        |10000     |12             | 
|DQN        |-100               |-20        |10000     |8              |
|DDPG       |43                 |280        |2000      |100            |
|TD3        |301                |305        |1073      |9              |

## Dependencies
Trained and tested on:
Python 3.8
gym 0.17.2
box2d 2.3.10
torch 1.5.0
numpy 1.18.4
Matplotlib 3.2.1
Pillow 7.1.2

## References
- DQN thesis (http://www.cs.rhul.ac.uk/~chrisw/thesis.html) and code (https://github.com/udacity/deep-reinforcement-learning)
- DDPG paper (https://arxiv.org/abs/1509.02971) and code (https://github.com/udacity/deep-reinforcement-learning)
- TD3 paper (https://arxiv.org/abs/1802.09477) and code (https://github.com/sfujim/TD3)

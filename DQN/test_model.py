
from DQN_TF.dqn import QNetwork
import torch
import gym

state_size = 14
action_size = 4

gym.logger.set_level(40)
env = gym.make('BipedalWalker-v3')
env.seed(0)

state = env.reset()[0:state_size]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Q-Network
qnetwork_local = QNetwork(state_size, action_size, 1).to(device)
qnetwork_local.forward(state)
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pickle
import torch


gym.logger.set_level(40)
env = gym.make('BipedalWalker-v3')
env.seed(0)
n_state_params = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

BATCH_SIZE = 64

MAX_EPISODES = 10
MAX_REWARD = 300
MAX_STEPS = 2000 #env._max_episode_steps
BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
MEAN_EVERY = 2

eps = 0.99
EPSILON_DECAY = 0.001
EPSILON_MIN = 0.001

start_episode = 0

agent = Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)

DIR = 'trained_models/'
LOAD = False
if LOAD:
    agent.epsilon = 0.001
    start_episode = 67300
    agent.qnetwork_local.load_state_dict(torch.load(DIR + 'checkpoint_local_ep' + str(start_episode) + '.pth', map_location="cpu"))
    agent.qnetwork_target.load_state_dict(torch.load(DIR + 'checkpoint_target_ep' + str(start_episode) + '.pth', map_location="cpu"))

scores = []
mean_scores = []
last_scores = deque(maxlen=MEAN_EVERY)
distances = []
mean_distances = []
last_distance = deque(maxlen=MEAN_EVERY)
losses_mean_episode = []

for ep in range(start_episode + 1, MAX_EPISODES + 1):
    state = env.reset()
    total_reward = 0
    total_distance = 0
    losses = []
    for t in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action, eps)
        loss = agent.step(state, action, reward, next_state, done)
        if loss is not None:
            losses.append(loss)
        state = next_state
        total_reward += reward
        if reward != -100:
            total_distance += reward
        if done:
            break
    eps = max(EPSILON_MIN, EPSILON_DECAY * eps)

    if len(losses) >= 1:
        mean_loss = np.mean(losses)
        losses_mean_episode.append((ep, mean_loss))
    else:
        mean_loss = None

    print('\rEpisode: {}/{},\tScore: {:.2f},\tDistance: {:.2f},\tloss: {},\te:{:.2f}'.format(ep, MAX_EPISODES,
                                                                                         total_reward,
                                                                                         total_distance, mean_loss,
                                                                                         agent.epsilon), end="")

    scores.append(total_reward)
    distances.append(total_distance)
    last_scores.append(total_reward)
    last_distance.append(total_distance)
    mean_score = np.mean(last_scores)
    mean_distance = np.mean(last_distance)

    # record rewards dynamically
    FILE = 'record.dat'
    data = [ep, total_reward, total_distance, mean_loss, agent.epsilon]
    with open(FILE, "ab") as f:
        pickle.dump(data, f)

    if (mean_score >= 300):
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, mean_score))
        torch.save(agent.qnetwork_local.state_dict(), DIR + '/best/checkpoint_local_ep' + str(ep) + '.pth')
        torch.save(agent.qnetwork_target.state_dict(), DIR + '/best/checkpoint_target_ep' + str(ep) + '.pth')
        break

    # save model every MEAN_EVERY episodes
    if ((ep % MEAN_EVERY) == 0):
        print('\rEpisode: {}/{},\tMean Score: {:.2f},\tMean Distance: {:.2f},\tloss: {},\te:{:.2f}'.format(ep, MAX_EPISODES,
                                                                                   mean_score,
                                                                                   mean_distance, mean_loss,
                                                                                   agent.epsilon))
        torch.save(agent.qnetwork_local.state_dict(), DIR + '/checkpoint_local_ep' + str(ep) + '.pth')
        torch.save(agent.qnetwork_target.state_dict(), DIR + '/checkpoint_target_ep' + str(ep) + '.pth')
        mean_scores.append(mean_score)
        mean_distances.append(mean_distance)
        FILE = 'record_mean.dat'
        data = [ep, mean_score, mean_distance, mean_loss, agent.epsilon]
        with open(FILE, "ab") as f:
            pickle.dump(data, f)
env.close()

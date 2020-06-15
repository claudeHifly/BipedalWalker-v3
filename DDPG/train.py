import gym
import torch
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt
import pickle
from collections import deque

gym.logger.set_level(40)
env = gym.make('BipedalWalker-v3')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

MAX_EPISODES = 20000
MAX_REWARD = 300
MAX_STEPS = 2000  # env._max_episode_steps
MEAN_EVERY = 100

start_episode = 0

agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0)

LOAD = True
noise = 1

if LOAD:
    start_episode = 10000
    agent.actor_local.load_state_dict(torch.load('./actor/checkpoint_actor_ep10000.pth', map_location="cpu"))
    agent.critic_local.load_state_dict(torch.load('./critic/checkpoint_critic_ep10000.pth', map_location="cpu"))
    agent.actor_target.load_state_dict(torch.load('./actor/checkpoint_actor_t_ep10000.pth', map_location="cpu"))
    agent.critic_target.load_state_dict(torch.load('./critic/checkpoint_critic_t_ep10000.pth', map_location="cpu"))

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
    actor_losses = []
    critic_losses = []
    for t in range(MAX_STEPS):

        # env.render()

        action = agent.act(state, noise)
        next_state, reward, done, info = env.step(action[0])
        actor_loss, critic_loss = agent.step(state, action, reward, next_state, done)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
        if critic_loss is not None:
            critic_losses.append(critic_loss)
        state = next_state.squeeze()
        state = next_state
        total_reward += reward
        if reward != -100:
            total_distance += reward
        if done:
            break

    if len(actor_losses) >= 1 and len(critic_losses) >= 1:
        mean_loss_actor = np.mean(actor_losses)
        mean_loss_critic = np.mean(critic_losses)
        losses_mean_episode.append((ep, mean_loss_actor, mean_loss_critic))
    else:
        mean_loss_actor = None
        mean_loss_critic = None

    print(
        '\rEpisode: {}/{},\tScore: {:.2f},\tDistance: {:.2f},\tactor_loss: {},\tcritic_loss:{}'.format(ep, MAX_EPISODES,
                                                                                                       total_reward,
                                                                                                       total_distance,
                                                                                                       mean_loss_actor,
                                                                                                       mean_loss_critic),
        end="")

    scores.append(total_reward)
    distances.append(total_distance)
    last_scores.append(total_reward)
    last_distance.append(total_distance)
    mean_score = np.mean(last_scores)
    mean_distance = np.mean(last_distance)
    FILE = 'record.dat'
    data = [ep, total_reward, total_distance, mean_loss_actor, mean_loss_critic]
    with open(FILE, "ab") as f:
        pickle.dump(data, f)

    if mean_score >= 300:
        print('Task Solved')
        torch.save(agent.actor_local.state_dict(), './actor/checkpoint_actor_best_ep' + str(ep) + '.pth')
        torch.save(agent.critic_local.state_dict(), './critic/checkpoint_critic_best_ep' + str(ep) + '.pth')
        torch.save(agent.actor_target.state_dict(), './actor/checkpoint_actor_best_t_ep' + str(ep) + '.pth')
        torch.save(agent.critic_target.state_dict(), './critic/checkpoint_critic_best_t_ep' + str(ep) + '.pth')
        break

    if ((ep % MEAN_EVERY) == 0):
        torch.save(agent.actor_local.state_dict(), './actor/checkpoint_actor_ep' + str(ep) + '.pth')
        torch.save(agent.critic_local.state_dict(), './critic/checkpoint_critic_ep' + str(ep) + '.pth')
        torch.save(agent.actor_target.state_dict(), './actor/checkpoint_actor_t_ep' + str(ep) + '.pth')
        torch.save(agent.critic_target.state_dict(), './critic/checkpoint_critic_t_ep' + str(ep) + '.pth')
        mean_scores.append(mean_score)
        mean_distances.append(mean_distance)
        print('\rEpisode: {}/{},\tMean Score: {:.2f},\tMean Distance: {:.2f},\tactor_loss: {},\tcritic_loss:{}'.format(
            ep, MAX_EPISODES,
            mean_score,
            mean_distance, mean_loss_actor,
            mean_loss_critic))
        FILE = 'record_mean.dat'
        data = [ep, mean_score, mean_distance, mean_loss_actor, mean_loss_critic]
        with open(FILE, "ab") as f:
            pickle.dump(data, f)
env.close()


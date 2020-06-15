
import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
from collections import deque
import pickle


######### Hyperparameters #########
gym.logger.set_level(40)
env_name = "BipedalWalker-v3"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

log_interval = 100  # print avg reward after interval
random_seed = 0
gamma = 0.99  # discount for future rewards
batch_size = 100  # num of transitions sampled from replay buffer
lr = 0.001
exploration_noise = 0.1
polyak = 0.995  # target policy update parameter (1-tau)
policy_noise = 0.2  # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2  # delayed policy updates parameter
max_episodes = 10000  # max num of episodes
max_timesteps = 2000  # max timesteps in one episode
directory = "./preTrained/"  # save trained models
filename = "TD3_{}_{}".format(env_name, random_seed)

start_episode = 0


policy = TD3(lr, state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

if random_seed:
    print("Random Seed: {}".format(random_seed))
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

LOAD = False
if LOAD:
    start_episode = 6
    policy.load(directory, filename, str(start_episode))

# logging variables:
scores = []
mean_scores = []
last_scores = deque(maxlen=log_interval)
distances = []
mean_distances = []
last_distance = deque(maxlen=log_interval)
losses_mean_episode = []

# training procedure:
for ep in range(start_episode + 1, max_episodes + 1):
    state = env.reset()
    total_reward = 0
    total_distance = 0
    actor_losses = []
    c1_losses = []
    c2_losses = []
    for t in range(max_timesteps):
        # select action and add exploration noise:
        action = policy.select_action(state)
        action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
        action = action.clip(env.action_space.low, env.action_space.high)

        # take action in env:
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, float(done)))
        state = next_state

        total_reward += reward
        if reward != -100:
            total_distance += reward

        # if episode is done then update policy:
        if done or t == (max_timesteps - 1):
            actor_loss, c1_loss, c2_loss = policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            actor_losses.append(actor_loss)
            c1_losses.append((c1_loss))
            c2_losses.append(c2_loss)
            break
    mean_loss_actor = np.mean(actor_losses)
    mean_loss_c1 = np.mean(c1_losses)
    mean_loss_c2 = np.mean(c2_losses)
    losses_mean_episode.append((ep, mean_loss_actor, mean_loss_c1, mean_loss_c2))
    print('\rEpisode: {}/{},\tScore: {:.2f},\tDistance: {:.2f},\tactor_loss: {},\tc1_loss:{},\tc2_loss:{}'
        .format(ep, max_episodes,total_reward,total_distance,mean_loss_actor,mean_loss_c1, mean_loss_c2),end="")

    # logging updates:
    scores.append(total_reward)
    distances.append(total_distance)
    last_scores.append(total_reward)
    last_distance.append(total_distance)
    mean_score = np.mean(last_scores)
    mean_distance = np.mean(last_distance)
    FILE = 'record.dat'
    data = [ep, total_reward, total_distance, mean_loss_actor, mean_loss_c1, mean_loss_c2]
    with open(FILE, "ab") as f:
        pickle.dump(data, f)

    # if avg reward > 300 then save and stop traning:
    if (mean_score) >= 300:
        print("########## Solved! ###########")
        name = filename + '_solved'
        policy.save(directory, name, str(ep))
        break

    # print avg reward every log interval:
    if ep % log_interval == 0:
        policy.save(directory, filename, str(ep))
        mean_scores.append(mean_score)
        mean_distances.append(mean_distance)
        print('\rEpisode: {}/{},\tMean Score: {:.2f},\tMean Distance: {:.2f},\tactor_loss: {},\tc1_loss:{},\tc2_loss:{}'
            .format(ep, max_episodes, mean_score, mean_distance, mean_loss_actor, mean_loss_c1, mean_loss_c2))
        FILE = 'record_mean.dat'
        data = [ep, mean_score, mean_distance, mean_loss_actor, mean_loss_c1, mean_loss_c2]
        with open(FILE, "ab") as f:
            pickle.dump(data, f)
env.close()

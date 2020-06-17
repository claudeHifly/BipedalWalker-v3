import gym
from TD3 import TD3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


gym.logger.set_level(40)
env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 100
lr = 0.002
max_timesteps = 2000
render = False
save_gif = False

filename = "TD3_{}_{}".format(env_name, random_seed)
filename += '_solved'
directory = "./preTrained/".format(env_name)
episode = 1073

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(lr, state_dim, action_dim, max_action)

policy.load_actor(directory, filename, episode)

scores = []

for ep in range(1, n_episodes+1):
    ep_reward = 0
    state = env.reset()
    for t in range(max_timesteps):
        action = policy.select_action(state)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
        if render:
            env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))
        if done:
            break
    scores.append(ep_reward)
    #print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    env.close()


print("Score media", np.mean(scores))
    
# plot
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(scores) + 1)
y = scores
plt.plot(x, y)
plt.grid()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Test Scores on BipedalWalker-v3')
plt.show()
namefig = "plot/test_scores_TD3_episodes" + str(episode) + ".jpg"
fig.savefig(namefig, dpi=300)
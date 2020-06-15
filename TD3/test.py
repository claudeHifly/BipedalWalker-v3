import gym
from TD3 import TD3
from PIL import Image


gym.logger.set_level(40)
env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 3
lr = 0.002
max_timesteps = 2000
render = True
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

    print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    env.close()


    
    

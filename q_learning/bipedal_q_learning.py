
import gym
import numpy as np
import random
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt

gym.logger.set_level(40)
env = gym.make('BipedalWalker-v3')



bucket_size_states = (4,5,5,5,4,5,4,5,2,4,5,4,5,2)
dim_states = len(bucket_size_states)

bucket_size_action = (20,20,20,20)
dim_action = len(bucket_size_action)
sBounds = [(0, math.pi),
           (-2,2),
           (-1,1),
           (-1,1),
           (0,math.pi),
           (-2,2),
           (0, math.pi),
           (-2,2),
           (0,1),
           (0, math.pi),
           (-2, 2),
           (0, math.pi),
           (-2, 2),
           (0, 1)]
aBounds = (-1, 1)

def update_Q(lear_rate, disc_rate, q_table, state, action, reward, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = q_table[state][action]  # estimate in Q-table (for current state, action pair)
    # get value of state, action pair at next time step
    Qsa_next = np.max(q_table[next_state]) if next_state is not None else 0
    target = reward + (disc_rate * Qsa_next)               # construct TD target
    new_value = current + (lear_rate * (target - current)) # get updated value
    return new_value

def state_to_bucket(state):
    bucket_state = []
    for i in range(len(state)):
        bucket_index = int((state[i]-sBounds[i][0])
                           / (sBounds[i][1]-sBounds[i][0])*bucket_size_states[i]-1)
        bucket_state.append(bucket_index)
    return tuple(bucket_state)



def bucket_to_action(bucket_action):
    action = []
    for i in range(len(bucket_action)):
        value_action = bucket_action[i] \
                       / (bucket_size_action[i] -1 ) * (aBounds[1] - aBounds[0]) - 1
        action.append(value_action)
    return tuple(action)



def choose_action(q_table, state, eps):
    # Select a random action
    if random.random() < eps:
        #print("azione random")
        action = ()
        for i in range (0, dim_action):
            action += (random.randint(0, bucket_size_action[i]-1),)
        #action = action_to_bucket(env.action_space.sample())
    # Select the action with the highest q
    else:

        action = np.unravel_index(np.argmax(q_table[state]), q_table[state].shape)
        #print("azione max q_table", action, "q-value", q_table[state][action])
    return action

def dd():
    return np.zeros(bucket_size_action)

def q_learning(env, num_episodes=100000, learn_rate= 0.01, disc_rate = 0.99, plot_every=1000):
    fig= plt.figure()
    xdata, ydata = [],[]
    ax=fig.add_subplot()
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    Ln, = ax.plot(xdata,ydata)
    ax.set_xlim([plot_every,num_episodes])
    ax.set_ylim([-100,300])


    #q_table = np.zeros(bucket_size_states + bucket_size_action)
    #q_table = defaultdict(lambda: np.zeros(bucket_size_action))
    q_table = defaultdict(dd)
    #monitor performance
    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes


    for i_episode in range(1, num_episodes + 1):
        state = state_to_bucket(env.reset()[0:dim_states])
        total_reward=0
        eps = 1.0/i_episode
        while True:
            #env.render()
            action = choose_action(q_table, state, eps)
            action_real = bucket_to_action(action)
            next_state_real, reward, done, info = env.step(action_real)
            next_state = state_to_bucket(next_state_real[0:dim_states])
            total_reward += reward
            q_table[state][action] = update_Q(learn_rate, disc_rate, q_table, state, action, reward, next_state)
            state = next_state
            if done:
                tmp_scores.append(total_reward)
                break
        if (i_episode % plot_every == 0):
            # plot performance
            print("salvo performance")
            xdata.append(i_episode)
            ydata.append(np.mean(tmp_scores))
            Ln.set_ydata(ydata)
            Ln.set_xdata(xdata)
            fig.show()
            fig.savefig("/content/drive/My Drive/Colab Notebooks/img_q_learning_4.png")
            # print best 100-episode performance
            #print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
            #f = open("/content/drive/My Drive/Colab Notebooks/q_table_learning_3.pkl", "wb")
            #pickle.dump(q_table,f)
            #f.close()
        print("num_episodio", i_episode)

    env.close()

    # plot performance


    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return q_table

Q_learn = q_learning(env)





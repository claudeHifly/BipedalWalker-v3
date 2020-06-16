
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

data = []
with open('./record.dat', 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass
data = pd.DataFrame(np.array(data))

scores = []
distances = []
actor_losses = []
critic1_losses = []
critic2_losses = []


for r in data[1]:
    scores.append(r)

for r in data[2]:
    distances.append(r)

for r in data[3]:
    actor_losses.append(r)

for r in data[4]:
    critic1_losses.append(r)

for r in data[4]:
    critic2_losses.append(r)

data_mean = []
with open('./record_mean.dat', 'rb') as record_mean:
    try:
        while True:
            data_mean.append(pickle.load(record_mean))
    except EOFError:
        pass
data_mean = pd.DataFrame(np.array(data_mean))

episode = []
scores_mean = []
distances_mean = []

for r in data_mean[0]:
    episode.append(r)

for r in data_mean[1]:
    scores_mean.append(r)

for r in data_mean[2]:
    distances_mean.append(r)


# plot
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(scores) + 1)
y = scores
plt.plot(x, y)
plt.grid()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Scores on BipedalWalker-v3')
plt.show()
namefig = "plot/scores_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)

# plot the distance and mean
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(distances) + 1)
y = distances
plt.plot(x, y)
plt.grid()
plt.ylabel('Distance')
plt.xlabel('Episode #')
plt.title('Distances on BipedalWalker-v3')
plt.show()
namefig = "plot/distances_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)



# plot the loss
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(actor_losses) + 1)
y = actor_losses
plt.plot(x, y)

plt.grid()
plt.ylabel('Actor Loss')
plt.xlabel('Episode #')
plt.title('Actor Loss on BipedalWalker-v3')
plt.show()
namefig = "plot/actor_loss_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(critic1_losses) + 1)
y = critic1_losses
plt.plot(x, y)

plt.grid()
plt.ylabel('Critic1 Loss')
plt.xlabel('Episode #')
plt.title('Critic Loss on BipedalWalker-v3')
plt.show()
namefig = "plot/critic1_loss_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(critic2_losses) + 1)
y = critic2_losses
plt.plot(x, y)

plt.grid()
plt.ylabel('Critic2 Loss')
plt.xlabel('Episode #')
plt.title('Critic Loss on BipedalWalker-v3')
plt.show()
namefig = "plot/critic2_loss_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)



# plot
fig = plt.figure()
ax = fig.add_subplot(111)
x = episode
y = scores_mean
plt.plot(x, y)
plt.grid()
plt.ylabel('Mean Scores every 100 episodes')
plt.xlabel('Episode #')
plt.title('Mean Scores on BipedalWalker-v3')
plt.show()
namefig = "plot/mean_scores_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)

# plot the distance and mean
fig = plt.figure()
ax = fig.add_subplot(111)
x = episode
y = distances_mean
plt.plot(x, y)
plt.grid()
plt.ylabel('Mean Distances every 100 episodes')
plt.xlabel('Episode #')
plt.title('Mean Distances on BipedalWalker-v3')
plt.show()
namefig = "plot/mean_distances_TD3_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)


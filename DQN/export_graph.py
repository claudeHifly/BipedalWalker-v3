
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

data = []
with open('record.dat', 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass
data = pd.DataFrame(np.array(data))

scores = []
distances = []
losses = []



for r in data[1]:
    scores.append(r)

for r in data[2]:
    distances.append(r)

for r in data[3]:
    losses.append(r)




# plot
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(scores) + 1)
y = scores
plt.plot(x, y)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Scores on BipedalWalker-v3')
plt.show()
namefig = "plot/scores_improved_DQN_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)

# plot the distance and mean
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(distances) + 1)
y = distances
plt.plot(x, y)
plt.ylabel('Distance')
plt.xlabel('Episode #')
plt.title('Distances on BipedalWalker-v3')
plt.show()
namefig = "plot/distances_improved_DQN_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)



# plot the loss
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(losses) + 1)
y = losses
plt.plot(x, y)
plt.ylabel('Agent Loss')
plt.xlabel('Episode #')
plt.title('Agent Loss on BipedalWalker-v3')
plt.show()
namefig = "plot/loss_improved_DQN_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)


data_mean = []
with open('record_mean.dat', 'rb') as record_mean:
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
x = episode
y = scores_mean
plt.plot(x, y)
plt.ylabel('Mean Scores every 100 episodes')
plt.xlabel('Episode #')
plt.title('Mean Scores on BipedalWalker-v3')
plt.show()
namefig = "plot/mean_scores_improved_DQN_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)

# plot the distance and mean
fig = plt.figure()
ax = fig.add_subplot(111)
x = episode
y = distances_mean
plt.plot(x, y)
plt.ylabel('Mean Distances every 100 episodes')
plt.xlabel('Episode #')
plt.title('Mean Distances on BipedalWalker-v3')
plt.show()
namefig = "plot/mean_distances_improved_DQN_episodes" + str(data.shape[0]) + ".jpg"
fig.savefig(namefig, dpi=300)


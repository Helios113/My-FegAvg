import torch
import numpy as np
import matplotlib.pyplot as plt


# rounds
rounds = 100
n = 3

# metrics = np.array(["Accuracy","Recall", "Precision", "F1", "Loss"])

train_p1 = torch.load("test_results/test_g/test_004/test_data")

train_p2 = torch.load("test_results/test_g/test_003/test_data")


###
# 1.
###
metric = 3
fig, axs = plt.subplots(1)

for i in train_p1.keys():
    i = 1
    x = train_p1[i][metric].reshape(rounds, -1)
    # print(metric)
    x_mean = x.mean(axis=1)
    x_err = 1.960 * x.std(axis=1) / np.sqrt(x.shape[1])
    axs.plot(x_mean.T, label="Set 1")
    axs.fill_between(range(rounds), x_mean - x_err, x_mean + x_err, alpha=0.2)
    # metric=metric+1
    break


for i in train_p2.keys():
    i = 1
    x = train_p2[i][metric].reshape(rounds, -1)
    x_mean = x.mean(axis=1)
    x_err = 1.960 * x.std(axis=1) / np.sqrt(x.shape[1])
    axs.plot(x_mean.T, "--", label="Set 2")
    axs.fill_between(range(rounds), x_mean - x_err, x_mean + x_err, alpha=0.2)
    break


axs.legend(loc=4, prop={"size": 20})
axs.set_ylim([0, 1])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

axs.set_ylabel("F1 Score", fontsize=20)
axs.set_xlabel("Global rounds", fontsize=20)
fig.tight_layout()
axs.patch.set_alpha(1)
plt.show()

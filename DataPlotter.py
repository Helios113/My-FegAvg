# printing performance
import torch
import numpy as np
import matplotlib.pyplot as plt


# rounds
rounds = 100
n = 4


# metrics = np.array(["Accuracy","Recall", "Precision", "F1", "Loss"])

train_p1 = torch.load("test_results/test_g/test_004/test_data")

train_p2 = torch.load("test_results/test_g/test_003/test_data")


###
# 1.
###
metric = 1
plt.rcParams["figure.figsize"] = (10, 10)
fig, axs = plt.subplots(1)
for i in train_p1.keys():
    # i=1
    x = train_p1[i][metric].reshape(rounds, -1)
    x_mean = x.mean(axis=1)
    x_err = 1.960 * x.std(axis=1)/np.sqrt(x.shape[1])
    axs.plot(x_mean.T, label="mm")
    # axs.fill_between(range(rounds), x_mean-x_err, x_mean+x_err,  alpha=0.2)
    # break

# rounds =
for i in train_p2.keys():
    # i=1
    x = train_p2[i][metric].reshape(rounds, -1)
    x_mean = x.mean(axis=1)
    x_err = 1.960 * x.std(axis=1)/np.sqrt(x.shape[1])
    axs.plot(x_mean.T, "--", label="single")
    # break
    # axs.fill_between(range(rounds), x_mean-x_err, x_mean+x_err,  alpha=0.2)


axs.legend()

# axs.set_ylim([0, 1])
axs.set(xlabel='Global rounds')
plt.show()


###
# 2.
###

# metric = [0]

# plt.rcParams["figure.figsize"] = (10,10)
# fig, axs = plt.subplots(n, sharex=True, sharey=True)
# plt.suptitle(f'{metrics[metric]} per device')
# for i in range(n):
#     # axs[i].plot(np.mean(train_performance[i,metric].reshape(len(metric),-1, tdw), axis=2).T, label=metrics[metric])
#     axs[i].plot(np.mean(train_performance[i,metric].reshape(len(metric),-1, tdw),axis=2).T, label=metrics[metric])
#     axs[i].legend()
# for ax in fig.get_axes():
#     ax.label_outer()
#     ax.set(ylabel = f'dev{fig.get_axes().index(ax)}')
#     if fig.get_axes()[-1]==ax:
#         ax.set(xlabel='Global rounds')

# plt.show()

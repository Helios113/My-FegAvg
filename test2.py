import numpy as np
import torch
import random
import pandas as pd

dataset = np.ones((100, 1))
targets = np.arange(4).repeat(25, 0).reshape(100, 1)
nDevices = 7
volPerClient = 1
classesPerClient = 2
clients = 4


dataset = np.hstack((dataset, targets))
data = pd.DataFrame(dataset)

# Best way
unique_classes = data.iloc[:, -1].unique()
number_classes = len(unique_classes)
inds = {}
for c in unique_classes:
    inds[c] = list(data.loc[data.iloc[:, -1] == c].index)
data_size = len(dataset)
budgetPerClass = np.ceil(data_size / (nDevices * classesPerClient))
D = {}
for i in range(nDevices):
    D[i] = []
    budgetPerDevice = data_size // (nDevices - i)
    data_size -= budgetPerDevice
    k = random.randint(0, number_classes - 1)
    while budgetPerDevice > 0:
        t = int(min(budgetPerDevice, budgetPerClass, len(inds[k])))
        budgetPerDevice -= t
        B = np.flip(np.sort(random.sample(range(len(inds[k])), t)))
        for j in B:
            D[i].append(inds[k].pop(j))
        k = (k + 1) % number_classes
    print(D)

import numpy as np
import torch
import random

dataset = np.ones((100, 1))
targets = np.arange(4).repeat(25, 0)
shard_per_user = 10
num_users = 4
"""
Sample non-I.I.D client data from MNIST dataset
:param dataset:
:param num_users:
:return:
"""
dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}
# empty dict

idxs_dict = {}
for i in range(len(dataset)):
    label = torch.tensor(targets[i]).item()
    if label not in idxs_dict.keys():
        idxs_dict[label] = []
    idxs_dict[label].append(i)
# indeces per label - idx_dict

num_classes = len(np.unique(targets))
# number of classes
rand_set_all = []
shard_per_class = int(shard_per_user * num_users / num_classes)
# shard per class - shard I guess its a sampling technique


for label in idxs_dict.keys():
    # for each unique label

    x = idxs_dict[label]

    num_leftover = len(x) % shard_per_class
    # how many elemnts are left after each shard

    leftover = x[-num_leftover:] if num_leftover > 0 else []
    # leftover
    x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
    # remove leftover from data
    x = x.reshape((shard_per_class, -1))
    x = list(x)

    for i, idx in enumerate(leftover):
        x[i] = np.concatenate([x[i], [idx]])
        # add leftover to the end of each matrix
    idxs_dict[label] = x

if len(rand_set_all) == 0:
    rand_set_all = list(range(num_classes)) * shard_per_class
    random.shuffle(rand_set_all)
    rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

# divide and assign
for i in range(num_users):
    rand_set_label = rand_set_all[i]
    rand_set = []
    for label in rand_set_label:
        idx = np.random.choice(len(idxs_dict[label]), replace=False)
        rand_set.append(idxs_dict[label].pop(idx))
    dict_users[i] = np.concatenate(rand_set)

test = []
for key, value in dict_users.items():
    x = np.unique(torch.tensor(dataset.targets)[value])
    assert (len(x)) <= shard_per_user
    test.append(value)
test = np.concatenate(test)
assert len(test) == len(dataset)
assert len(set(list(test))) == len(dataset)

import torch
from torch import nn
from torch.utils.data import Dataset
from MHealthDataset import MHealthDataset
import numpy as np
import random
random.seed(0)
np.random.seed(0)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset[0]
        self.labels = dataset[1]
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return (torch.Tensor(self.dataset[self.idxs[item]]), int(self.labels[self.idxs[item]]))


def get_data(path, clients):
    dataset_train, dataset_test = MHealthDataset(
        path, 0.8, 0.2)
    dict_users_train = iid_data(dataset_train, clients)
    dict_users_test = iid_data(dataset_test, clients)
    return dataset_train, dataset_test, dict_users_train, dict_users_test


def iid_data(dataset, num_users):
    num_items = int(dataset[0].shape[0]/num_users)
    dict_users, idxs = {}, [i for i in range(dataset[0].shape[0])]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            idxs, num_items, replace=False))
        idxs = list(set(idxs) - dict_users[i])
    return dict_users


import torch
from torch import nn
from torch.utils.data import Dataset
from MHealthDataset import MHealthDataset
import numpy as np
import pandas as pd
import random


class FedDataset(Dataset):
    def __init__(self, data, device="mps"):
        self.data = torch.Tensor(data[:, :-1]).to(device).unsqueeze(1)
        self.target = torch.LongTensor(data[:, -1]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [self.data[item], self.target[item]]


def get_data(path, clients, iid):
    dataset_train, dataset_test = load_data(path)
    if iid:
        dict_users_train = iid_data(dataset_train, clients)
        dict_users_test = iid_data(dataset_test, clients)
    else:
        dict_users_train = non_iid_data(dataset_train, clients, 4)
        dict_users_test = non_iid_data(dataset_test, clients, 4)
    return dict_users_train, dict_users_test


def iid_data(dataset, num_users):
    np.random.seed(10)
    num_items = int(dataset[0].shape[0] / num_users)
    dict_users, idxs = {}, [i for i in range(dataset[0].shape[0])]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(idxs, num_items, replace=False))
        idxs = list(set(idxs) - dict_users[i])
    return dict_users


def non_iid_data(data, num_users, classesPerClient):
    random.seed(10)
    unique_classes = np.unique(data[:, -1])
    number_classes = len(unique_classes)
    inds = {}
    for c in unique_classes:
        inds[c] = np.where(data[:, -1] == c)[0].tolist()
    data_size = len(data)
    budgetPerClass = np.ceil(data_size / (num_users * classesPerClient))
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = None
        budgetPerDevice = data_size // (num_users - i)
        data_size -= budgetPerDevice
        k = random.randint(0, number_classes - 1)
        while budgetPerDevice > 0:
            t = int(min(budgetPerDevice, budgetPerClass, len(inds[k])))
            budgetPerDevice -= t
            B = np.flip(np.sort(random.sample(range(len(inds[k])), t)))
            for j in B:
                if dict_users[i] is None:
                    dict_users[i] = data[inds[k].pop(j)]
                else:
                    dict_users[i] = np.vstack((dict_users[i], data[inds[k].pop(j)]))
            k = (k + 1) % number_classes
    return dict_users


def load_data(path):
    data = pd.read_csv(path, header=None, index_col=False)
    train = data.sample(frac=0.7, random_state=10)
    test = data.drop(train.index).values
    train = train.values
    return train, test

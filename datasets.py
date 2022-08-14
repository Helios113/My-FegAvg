from zoneinfo import available_timezones
import torch
from torch import nn
from torch.utils.data import Dataset
from MHealthDataset import MHealthDataset
from OPPDataset import OPPDataset
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


def get_data(path, iidtype, clients, share):
    dataset_train, dataset_test = MHealthDataset(
        path, 0.08, 0.02)  # ,transform=trans_mnist)
    if iidtype:
        dict_users_train = iid(dataset_train, clients)
        dict_users_test = iid(dataset_test, clients)
    else:
        dict_users_train, rand_set_all = noniid(dataset_train, clients, share)
        dict_users_test, rand_set_all = noniid(
            dataset_test, clients, share)
    return dataset_train, dataset_test, dict_users_train, dict_users_test


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    num_items = int(dataset[0].shape[0]/num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset[0].shape[0])]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(dataset[0].shape[0]):
        label = torch.tensor(dataset[1][i]).item()
        if label not in idxs_dict.keys():
            # for each class make an empty list
            idxs_dict[label] = []
        # add the location of each input in the appropriate class
        idxs_dict[label].append(i)

    # get number of classes
    num_classes = len(np.unique(dataset[1]))
    available_classes = np.unique(dataset[1])
    available_classes = available_classes.astype('int32')
    # shared per class
    shard_per_class = int(shard_per_user * num_users / num_classes)

    for label in idxs_dict.keys():
        # for each class

        # get all the datainputs for this class
        x = idxs_dict[label]

        # how many classes are leftover
        num_leftover = len(x) % shard_per_class

        #
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        # for class label assign x to be the data indexes
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(available_classes) * shard_per_class
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
        x = np.unique(torch.tensor(dataset[1])[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == dataset[0].shape[0])
    assert(len(set(list(test))) == dataset[0].shape[0])

    return dict_users, rand_set_all

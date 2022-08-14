import pickle
import copy
import torch
from client import Client
from models import SLC, MLP
from datasets import DatasetSplit, get_data
import numpy as np
import os
from torch.utils.data import DataLoader
import IPython.display as dsp
import matplotlib.pyplot as plt
from math import ceil
from win10toast import ToastNotifier
toaster = ToastNotifier()


rounds = 40
test_freq = 1
local_epochs = 2

# Test parameters
temporal_len = 10

transient_dim = 32
output_dim = 12
hidden_dims = [64]

batch_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train_data, test_data, train_dict, test_dict = get_data("data/data_all.csv", True , n,20)
# num_batch =  ceil(max([len(i) for i in test_dict.values()])/batch_size)
# print(num_batch)
# # most likely issue


all_mod = {
    "acc": [1, 2, 3, 6, 7, 8, 15, 16, 17, 9, 10, 11, 18, 19, 20, 12, 13, 14, 21, 22, 23],
    "acc33": [1, 2, 3, 6, 7, 8, 15, 16, 17],
    "gyro": [9, 10, 11, 18, 19, 20],
    "mag": [12, 13, 14, 21, 22, 23]
}


# Fill modalities
n = 3  # number of clients
modalities = [
    {
        "acc": [1, 2, 3, 6, 7, 8, 15, 16, 17, 9, 10, 11, 18, 19, 20, 12, 13, 14, 21, 22, 23],
    },
    {
        "acc": [1, 2, 3, 6, 7, 8, 15, 16, 17, 9, 10, 11, 18, 19, 20, 12, 13, 14, 21, 22, 23],
    },
    {
        "acc": [1, 2, 3, 6, 7, 8, 15, 16, 17, 9, 10, 11, 18, 19, 20, 12, 13, 14, 21, 22, 23],
    }
]

# Mode

federatedGlob = False
federatedLoc = False
lg_frac = 0

# result lists

train_performance = []
test_performance = np.zeros((n, 5, rounds))


def create_clients():
    clients = []
    if federatedLoc:
        uni_loc = SLC(all_mod,
                      hidden_dims, transient_dim, False)
    uni_glob = MLP(transient_dim, output_dim)
    # Generate clients
    for i in range(n):
        local_mod = SLC(modalities[i],
                        hidden_dims, transient_dim, False)
        # local_mod = MLP_CL(len(modalities[i])*temporal_len,transient_dim)
        # local_mod = LSTM(modalities[i], hidden_dims, transient_dim)
        glob_mod = MLP(transient_dim, output_dim)

        if federatedLoc:
            s_dict = {}
            local_dict = uni_loc.state_dict()
            for k in local_mod.state_dict():
                s_dict[k] = copy.deepcopy(local_dict[k])
            local_mod.load_state_dict(s_dict)

        s_dict = {}
        global_dict = uni_glob.state_dict()
        for k in glob_mod.state_dict():
            s_dict[k] = copy.deepcopy(global_dict[k])
        glob_mod.load_state_dict(s_dict)

        clients.append(
            Client(glob_mod, local_mod, local_epochs,
                   learning_rate=1e-2, device=device)
        )


def train():


for round in range(rounds):
    # Global params for FL
    w_glob_tmp = None

    # Local params for FL
    w_loc_tmp = None

    # Count of encounters of each param
    w_loc_tmp_count = None
    if round > (1-lg_frac)*rounds:
        federatedLoc = False
    print_loss = []
    for client in range(n):
        w_glob_ret, w_local_ret, performance = clients[client].train(
            DataLoader(DatasetSplit(
                train_data, train_dict[0]), batch_size=32, shuffle=True)
        )
        print_loss.append(copy.deepcopy(np.average(performance)))

        if federatedGlob:
            # Global model, usually classifier
            # Always shared between all devices
            # All keys are the same on all clients
            if w_glob_tmp is None:
                w_glob_tmp = copy.deepcopy(w_glob_ret)
            else:
                for k in w_glob_ret:
                    w_glob_tmp[k] += w_glob_ret[k]

        if federatedLoc:
            # Local model, usually some encoder
            # Can have multiple modalities
            # Can be different between all clients
            if w_loc_tmp is None:
                w_loc_tmp = copy.deepcopy(w_local_ret)
                w_loc_tmp_count = dict.fromkeys(w_local_ret.keys(), 1)
            else:
                for k in w_local_ret.keys():
                    if k not in w_loc_tmp:
                        w_loc_tmp[k] = w_local_ret[k]
                        w_loc_tmp_count[k] = 1
                    else:
                        w_loc_tmp[k] += w_local_ret[k]
                        w_loc_tmp_count[k] += 1

        performance = clients[client].test(DataLoader(DatasetSplit(
            test_data, test_dict[0]), batch_size=32, shuffle=True))
        test_performance[client, :, round:(
            round+1)] = copy.deepcopy(np.average(performance, axis=1).reshape(5, 1))

    train_performance.append(np.average(print_loss))
    # get weighted average for global weights
    if federatedGlob:
        for k in w_glob_tmp.keys():
            w_glob_tmp[k] = torch.div(w_glob_tmp[k], n)
    if federatedLoc:
        for k in w_loc_tmp.keys():
            w_loc_tmp[k] = torch.div(w_loc_tmp[k], w_loc_tmp_count[k])

    # copy weights to each client based on mode
    if federatedGlob or federatedLoc:
        for client in range(n):
            clients[client].load_params(w_glob_tmp, w_loc_tmp)

    dsp.clear_output(wait=True)
    plt.clf()
    plt.plot(train_performance)
    dsp.display(plt.gcf())
    print(modalities)


save_path = "test_results/test_fed_multimodal"

if os.path.isdir(save_path) is False:
    os.mkdir(save_path)
    np.savez(f"{save_path}/test_data.npz",
             data=test_performance.reshape(n*5, -1), allow_pickle=True)
    for i in range(n):
        torch.save(clients[i].model.state_dict(), f"{save_path}/dev{i}_model")
else:
    print("test exists")

toaster.show_toast("Analysis Done")

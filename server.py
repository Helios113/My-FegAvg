import copy
import torch
from client import Client
from models import SLC, MLP
from dataset import FedDataset, get_data
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
from datetime import datetime
import yaml


torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description="DM-FedAvg")
# Optional argument
parser.add_argument(
    "--split", type=float, nargs="?", default=0.7, help="Training data portion"
)

parser.add_argument(
    "--paramPath",
    required=True,
    action="store",
    type=str,
    help="yaml file with modality parameters",
)

args = parser.parse_args()

dirs = os.path.split(args.paramPath)
save_path = dirs[0]
# model parameters
params_loc_path = os.path.join(save_path, "paramsLoc")
params_glob_path = os.path.join(save_path, "paramsGlob")
loss_train_path = os.path.join(save_path, "loss_train.txt")
loss_test_path = os.path.join(save_path, "loss_test.txt")
f1_test_path = os.path.join(save_path, "f1_test.txt")
info_path = os.path.join(save_path, "info.txt")


"""
Fiels
"""
if not os.path.exists(save_path):
    os.makedirs(save_path)

lss_train_f = open(loss_train_path, "w+")
lss_test_f = open(loss_test_path, "w+")

f1_test_f = open(f1_test_path, "w+")
info_f = open(info_path, "w+")
test_freq = 1


# Determine hardware availability
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available


# Test parameters


all_mod = {
    "all": [
        1,
        2,
        3,
        6,
        7,
        8,
        15,
        16,
        17,
        9,
        10,
        11,
        18,
        19,
        20,
        12,
        13,
        14,
        21,
        22,
        23,
    ],
    "acc": [1, 2, 3, 6, 7, 8, 15, 16, 17],
    "gyro": [9, 10, 11, 18, 19, 20],
    "mag": [12, 13, 14, 21, 22, 23],
}

try:
    with open(args.paramPath, "r") as file:
        yml_file = yaml.safe_load(file)
        modalities = yml_file["modalities"]
        transient_dim = 4
        output_dim = 13
        hidden_dims = [32]
        num_clients = yml_file["n_clients"]
        num_sets = yml_file["n_sets"]

        batch_size = yml_file["batch_size"]
        federatedGlob = yml_file["fedGlob"]
        federatedLoc = yml_file["fedLoc"]

        learning_rate = float(yml_file["lr"])
        momentum = 0
        optimizer = yml_file["optim"]

        alpha = 1
        alpha_per_modality = False
        lg_frac = yml_file["lg_frac"]
        rounds = yml_file["rounds"]
        local_epochs = yml_file["epochs"]

except:
    print("no yaml file given")
    exit()
# modalities = [
#     {
#         "all": [
#             1,
#             2,
#             3,
#             6,
#             7,
#             8,
#             15,
#             16,
#             17,
#             9,
#             10,
#             11,
#             18,
#             19,
#             20,
#             12,
#             13,
#             14,
#             21,
#             22,
#             23,
#         ],
#     },
#     {
#         "all": [
#             1,
#             2,
#             3,
#             6,
#             7,
#             8,
#             15,
#             16,
#             17,
#             9,
#             10,
#             11,
#             18,
#             19,
#             20,
#             12,
#             13,
#             14,
#             21,
#             22,
#             23,
#         ],
#     },
#     {
#         "all": [
#             1,
#             2,
#             3,
#             6,
#             7,
#             8,
#             15,
#             16,
#             17,
#             9,
#             10,
#             11,
#             18,
#             19,
#             20,
#             12,
#             13,
#             14,
#             21,
#             22,
#             23,
#         ],
#     },
#     {
#         "all": [
#             1,
#             2,
#             3,
#             6,
#             7,
#             8,
#             15,
#             16,
#             17,
#             9,
#             10,
#             11,
#             18,
#             19,
#             20,
#             12,
#             13,
#             14,
#             21,
#             22,
#             23,
#         ],
#     },
# ]

print(r"{:-^30}".format("PID"), file=info_f)
print(r"{txt:<20}:{val}".format(txt="pid", val=os.getpid()), file=info_f)
print(args, file=info_f)
print(modalities, file=info_f)
info_f.flush()

data_train, data_test = get_data(
    "/home/preslav/Projects/My-FegAvg/data/data_all.csv", 4, False
)

clients = []

if federatedLoc:
    uni_loc = SLC(all_mod, hidden_dims, transient_dim, False)
uni_glob = MLP(transient_dim, output_dim)

for i in range(num_clients):
    glob_mod = MLP(transient_dim, output_dim)
    local_mod = SLC(modalities[i], hidden_dims, transient_dim, False)
    if federatedLoc:
        s_dict = {}
        local_dict = uni_loc.state_dict()
        for k in local_mod.state_dict():
            s_dict[k] = copy.deepcopy(local_dict[k])
        local_mod.load_state_dict(s_dict)
    if federatedGlob:
        s_dict = {}
        global_dict = uni_glob.state_dict()
        for k in glob_mod.state_dict():
            s_dict[k] = copy.deepcopy(global_dict[k])
        glob_mod.load_state_dict(s_dict)

    clients.append(
        Client(
            glob_mod,
            local_mod,
            DataLoader(
                FedDataset(data_train[i % num_sets], device),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                FedDataset(data_test[i % num_sets], device),
                batch_size=batch_size,
                shuffle=True,
            ),
            local_epochs,
            learning_rate,
            optimizer,
            device=device,
        )
    )


last_entry = 0
performance = np.zeros((num_clients, 2, rounds))
loss = np.zeros((num_clients, 2, rounds))
init_time = datetime.now()
max_f1 = np.zeros(num_clients)
for round in range(rounds):
    last_time = datetime.now()
    # Global params for FL
    w_glob_tmp = None
    # Local params for FL
    w_loc_tmp = None

    # Count of encounters of each param
    w_loc_tmp_count = None
    if round > (1 - lg_frac) * rounds:
        federatedLoc = False

    for client in range(num_clients):
        w_glob_ret, w_local_ret, loss[client, 0, round] = clients[client].train()
        (
            performance[client, 0, round],
            performance[client, 1, round],
            loss[client, 1, round],
            _,
        ) = clients[client].test()

        if federatedGlob:
            if w_glob_tmp is None:
                w_glob_tmp = copy.deepcopy(w_glob_ret)
            else:
                for k in w_glob_ret:
                    w_glob_tmp[k] += w_glob_ret[k]

        if federatedLoc:
            if alpha_per_modality:
                factor = (
                    1 if len(w_local_ret) / 8 == 1 else len(w_local_ret) / 8 * alpha
                )
            else:
                factor = 1 if len(w_local_ret) / 8 == 1 else alpha

            if w_loc_tmp is None:
                w_loc_tmp = {}
                w_loc_tmp_count = {}
            for k in w_local_ret.keys():
                if k not in w_loc_tmp:
                    w_loc_tmp[k] = factor * w_local_ret[k]
                    w_loc_tmp_count[k] = factor
                else:
                    w_loc_tmp[k] += factor * w_local_ret[k]
                    w_loc_tmp_count[k] += factor

    for client in range(num_clients):
        if performance[client, 0, round] > max_f1[client]:
            max_f1[client] = performance[client, 0, round]
            torch.save(
                clients[client].get_params()[0], params_glob_path + f"{client}.mp"
            )
            torch.save(
                clients[client].get_params()[1], params_loc_path + f"{client}.mp"
            )

    train_loss = np.char.mod("%f", loss[:, 0, round].reshape(-1))
    train_loss = ",".join(train_loss)

    mean_std = np.char.mod("%f", performance[:, :, round].reshape(-1))
    mean_std = ",".join(mean_std)

    test_loss = np.char.mod("%f", loss[:, 1, round].reshape(-1))
    test_loss = ",".join(test_loss)
    print(
        r"{},{},{},{}".format(
            datetime.now() - init_time, datetime.now() - last_time, round, train_loss
        ),
        file=lss_train_f,
    )
    print(
        r"{},{},{},{} ".format(
            datetime.now() - init_time, datetime.now() - last_time, round, test_loss
        ),
        file=lss_test_f,
    )
    print(
        r"{},{},{},{} ".format(
            datetime.now() - init_time, datetime.now() - last_time, round, mean_std
        ),
        file=f1_test_f,
    )

    # get weighted average for global weights
    if federatedGlob:
        for k in w_glob_tmp.keys():
            w_glob_tmp[k] = torch.div(w_glob_tmp[k], num_clients)
    if federatedLoc:
        for k in w_loc_tmp.keys():
            w_loc_tmp[k] = torch.div(w_loc_tmp[k], w_loc_tmp_count[k])

    # copy weights to each client based on mode
    if federatedGlob or federatedLoc:
        for client in range(num_clients):
            clients[client].load_params(w_glob_tmp, w_loc_tmp)

    lss_train_f.flush()
    lss_test_f.flush()
    f1_test_f.flush()

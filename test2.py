import copy
import torch
from client import Client
from models import SLC, MLP
from dataset import FedDataset, get_data, DataInfo
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
        output_dim = 13

        yml_file = yaml.safe_load(file)
        modalities = yml_file["modalities"]
        transient_dim = yml_file["transient_dim"]

        hidden_dims = yml_file["hidden_dims"]
        num_clients = yml_file["n_clients"]
        num_sets_train = yml_file["n_sets_train"]
        num_sets_test = yml_file["n_sets_test"]

        batch_size = yml_file["batch_size"]
        federatedGlob = yml_file["fedGlob"]
        federatedLoc = yml_file["fedLoc"]

        learning_rate = float(yml_file["lr"])
        optimizer = yml_file["optim"]

        alpha = 1
        alpha_per_modality = False
        lg_frac = yml_file["lg_frac"]
        rounds = yml_file["rounds"]
        local_epochs = yml_file["epochs"]
        iid = yml_file["iid"]
        penalty = yml_file["penalty"]
        classes_per_client_training = yml_file["classes_per_client_training"]
        classes_per_client_testing = yml_file["classes_per_client_testing"]
        train_data_portion = yml_file["train_data_portion"]
        test_data_portion = yml_file["test_data_portion"]
except:
    print("no yaml file given")
    exit()

data_train, data_test = get_data(
    "/home/preslav/Projects/My-FegAvg/data/data_all.csv",
    num_clients,
    iid,
    DataInfo(
        train_data_portion,
        test_data_portion,
        classes_per_client_training,
        classes_per_client_testing,
        num_sets_train,
        num_sets_test,
    ),
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
        local_dict = torch.load(params_loc_path + f"{i}.mp")
        for k in local_mod.state_dict():
            s_dict[k] = copy.deepcopy(local_dict[k])
        local_mod.load_state_dict(s_dict)
    if federatedGlob:
        s_dict = {}
        global_dict = torch.load(params_glob_path + f"{i}.mp")
        for k in glob_mod.state_dict():
            s_dict[k] = copy.deepcopy(global_dict[k])
        glob_mod.load_state_dict(s_dict)

    clients.append(
        Client(
            glob_mod,
            local_mod,
            DataLoader(
                FedDataset(data_train[i % num_sets_train], device),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                FedDataset(data_test[i % num_sets_test], device),
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
last_time = datetime.now()
# Global params for FL
w_glob_tmp = None
# Local params for FL
w_loc_tmp = None

for client in range(num_clients):
    (
        performance[client, 0, 0],
        performance[client, 1, 0],
        loss[client, 1, 0],
        vec,
    ) = clients[client].test()
    print(vec[0])
    # break

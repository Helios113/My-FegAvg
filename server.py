import copy
import torch
from client import Client
from models import SLC, MLP
from dataset import FedDataset, get_data, DataInfo
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
import yaml


torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description="DM-FedAvg")
# Optional argument

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
parent_path = os.path.split(save_path)[0]
# model parameters
params_loc_path = os.path.join(save_path, "paramsLoc")
params_glob_path = os.path.join(save_path, "paramsGlob")
loss_train_path = os.path.join(save_path, "loss_train.txt")
loss_test_path = os.path.join(save_path, "loss_test.txt")
f1_test_path = os.path.join(save_path, "f1_test.txt")
acc_test_path = os.path.join(save_path, "acc_test.txt")

info_path = os.path.join(save_path, "info.txt")

glob_file = os.path.join(parent_path, "glob")
loc_file = os.path.join(parent_path, "loc")

"""
Fiels
"""
if not os.path.exists(save_path):
    os.makedirs(save_path)

lss_train_f = open(loss_train_path, "w+")
lss_test_f = open(loss_test_path, "w+")

f1_test_f = open(f1_test_path, "w+")
acc_test_f = open(acc_test_path, "w+")

info_f = open(info_path, "w+")
test_freq = 1


# Determine hardware availability
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

# device = "cpu"
# Test parameters


all_mod = {
    "all": [0, 1, 2, 5, 6, 7, 14, 15, 16, 8, 9, 10, 17, 18, 19, 11, 12, 13, 20, 21, 22],
    "acc": [0, 1, 2, 5, 6, 7, 14, 15, 16],
    "acc1": [0, 1, 2],
    "acc2": [5, 6, 7],
    "acc3": [14, 15, 16],
    "gyro": [8, 9, 10, 17, 18, 19],
    "gyro1": [8, 9, 10],
    "gyro2": [17, 18, 19],
    "mag": [ 11, 12, 13, 20, 21, 22],
    "mag1": [11, 12, 13,],
    "mag2": [ 20, 21, 22],
}
try:
    with open(args.paramPath, "r") as file:
        yml_file = yaml.safe_load(file)
        modalities = yml_file["modalities"]

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
        mlp_dims = yml_file["mlp_dims"]
        transient_dim = mlp_dims[0]
        output_dim=mlp_dims[-1]
        train = yml_file["train"]
        data_path = yml_file["data_path"]
except Exception as e:
    print(e)
except:
    print("no yaml file given")
    exit()

print(r"{:-^30}".format("PID"), file=info_f)
print(r"{txt:<20}:{val}".format(txt="pid", val=os.getpid()), file=info_f)
print(yml_file, file=info_f)
print(modalities, file=info_f)
print(datetime.now(), file=info_f)


data_train, data_test = get_data(
    data_path,
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
info_f.flush()
clients = []

uni_loc = SLC(all_mod, hidden_dims, transient_dim, penalty)
uni_glob = MLP(mlp_dims)
if os.path.exists(glob_file):
    uni_glob.load_state_dict(torch.load(glob_file))
else:
    torch.save(uni_glob.state_dict(),glob_file)
if os.path.exists(loc_file):
    uni_loc.load_state_dict(torch.load(loc_file))
else:
    torch.save(uni_loc.state_dict(),loc_file)
    
for i in range(num_clients):
    glob_mod = MLP(mlp_dims)
    local_mod = SLC(modalities[i], hidden_dims, transient_dim, penalty)
    
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
                batch_size=32,
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
acc = np.zeros((num_clients, 2, rounds))
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
    # if round > (1 - lg_frac) * rounds:
    #     federatedGlob = True

    for client in range(num_clients):
       
        w_glob_ret, w_local_ret, loss[client,
                                        0, round] = clients[client].train(train, round)
        (
            performance[client, 0, round],
            performance[client, 1, round],
            acc[client, 0, round],
            acc[client, 1, round],_
        ) = clients[client].test()

        if federatedGlob:
            if w_glob_tmp is None:
                w_glob_tmp = copy.deepcopy(w_glob_ret)
                for k in w_glob_ret:
                    w_glob_tmp[k] = w_glob_tmp[k].unsqueeze(0)
            else:
                for k in w_glob_ret:
                    w_glob_tmp[k] = torch.cat((w_glob_tmp[k], w_glob_ret[k].unsqueeze(0)), dim=0)

        if federatedLoc:
            if w_loc_tmp is None:
                w_loc_tmp = {}
                w_loc_tmp_count = {}
            for k in w_local_ret.keys():
                if k not in w_loc_tmp:
                    w_loc_tmp[k] =  w_local_ret[k]
                    w_loc_tmp_count[k] = 1
                else:
                    w_loc_tmp[k] +=  w_local_ret[k]
                    w_loc_tmp_count[k] += 1

    for client in range(num_clients):
        if performance[client, 0, round] > max_f1[client]:
            max_f1[client] = performance[client, 0, round]
            torch.save(
                clients[client].get_params()[0], params_glob_path +
                f"{client}.mp"
            )
            torch.save(
                clients[client].get_params()[1], params_loc_path +
                f"{client}.mp"
            )

    train_loss = np.char.mod("%f", loss[:, 0, round].reshape(-1))
    train_loss = ",".join(train_loss)

    mean_std = np.char.mod("%f", performance[:, :, round].reshape(-1))
    mean_std = ",".join(mean_std)
    
    mean_std1 = np.char.mod("%f", acc[:, :, round].reshape(-1))
    mean_std1 = ",".join(mean_std1)

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
    print(
        r"{},{},{},{} ".format(
            datetime.now() - init_time, datetime.now() - last_time, round, mean_std1
        ),
        file=acc_test_f,
    )

    # get weighted average for global weights
    if federatedGlob:
        for k in w_glob_tmp.keys():
            w_glob_tmp[k] = torch.mean(w_glob_tmp[k], 0, False).squeeze(0)
    if federatedLoc:
        for k in w_loc_tmp.keys():
            w_loc_tmp[k] = torch.div(w_loc_tmp[k], w_loc_tmp_count[k])

    # copy weights to each client based on mode
    if federatedGlob or federatedLoc:
        for client in range(num_clients):
            clients[client].load_params(w_glob_tmp, w_loc_tmp)

    if round%10==0:
        lss_train_f.flush()
        lss_test_f.flush()
        f1_test_f.flush()
        acc_test_f.flush()

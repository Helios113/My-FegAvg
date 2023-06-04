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


glob_file = "NEW_DATA_M/nazarene/glob"
loc_file = "NEW_DATA_M/nazarene/loc"

# Determine hardware availability
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
    
all_mod = {
    "acc1": [0, 1, 2],
    "acc2": [5, 6, 7],
    "gyro1": [8, 9, 10],
    "gyro2": [17, 18, 19],
    "mag2": [ 20, 21, 22],
}
hidden_dims = 2
transient_dim = 32
mlp_dims = [32, 16, 12]
uni_loc = SLC(all_mod, hidden_dims, transient_dim, True)
uni_glob = MLP(mlp_dims)


aa = uni_glob.state_dict()



a = torch.load(glob_file)
uni_glob.load_state_dict(a)
b = torch.load(loc_file)
s_dict = {}

for k in (uni_loc.get_submodule("layer_list").state_dict()):
    s_dict[k] = b[k]
uni_loc.get_submodule("layer_list").load_state_dict(s_dict)
bb = uni_loc.state_dict()
c = 2
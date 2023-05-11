import re
from tkinter.messagebox import NO
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

torch.manual_seed(0)
np.random.seed(0)


class Composite(nn.Module):
    def __init__(self, local: nn.Module, glob: nn.Module):
        super(Composite, self).__init__()
        self.local = local
        self.glob = glob

    def forward(self, x):
        x = self.local(x)
        return self.glob(x), x


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(LSTM, self).__init__()

        input_dim = np.array(input_dim).flatten()
        self.layers = [len(input_dim)] + hidden_dims + [output_dim]
        self.layer_size = len(self.layers) - 1
        self.layer_list = nn.ModuleList()
        for i in range(self.layer_size):
            self.layer_list.append(
                nn.LSTM(
                    input_size=self.layers[i],
                    hidden_size=self.layers[i + 1],
                    batch_first=True,
                )
            )

    def forward(self, x):
        h_n = None
        for i, layer in enumerate(self.layer_list):
            x, (h_n, c_n) = layer(x)
        return h_n


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Dropout())    
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.linear(x)
        return x


class MLP_drop(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_drop, self).__init__()
        self.seq = nn.Sequential(nn.Dropout(0.4), nn.Linear(input_dim, output_dim))

    def forward(self, x):
        x = x.squeeze()
        x = self.seq(x)
        return x


class MLP_CL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_CL, self).__init__()
        self.input_dim = input_dim
        self.stack = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Dropout(p=0.4),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.stack(x)


class SLC(nn.Module):
    def __init__(self, modalities: dict, hidden_dims, output_dim, penalty):
        super(SLC, self).__init__()
        self.penalty = penalty
        self.modalities = modalities
        self.layer_list = nn.ModuleDict()
        self.output_dim = output_dim
        for i in modalities:
            self.layer_list[i] = nn.LSTM(
                        input_size = len(modalities[i]),
                        hidden_size = output_dim,
                        num_layers = hidden_dims,
                        batch_first=True,
                        dropout = 0.5
                    )

    def forward(self, x):
        h = None
        h_list = []
        penalty = None
        size = len(self.modalities)
        for i in self.modalities:
            X = x[..., self.modalities[i]]
            X, (h_n, c_n) = self.layer_list[i](X)
            h_n=h_n[-1]
            h_list.append(h_n)
            if h == None:
                h = h_n / size
            else:
                h += h_n / size
        if self.penalty:
            for comb in combinations(h_list, 2):
                if penalty is None:
                    penalty = torch.abs(comb[0] - comb[1])
                else:
                    penalty += torch.abs(comb[0] - comb[1])
            if penalty is None:
                penalty = torch.zeros_like(h)
            return h - penalty
        return h

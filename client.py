
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from copy import copy
from time import sleep
from models import Composite, LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Client:
    def __init__(self, glob, local, epochs, learning_rate, momentum, optimizer, device):
        self.device = device
        self.momentum = momentum
        self.optim = optimizer
        glob.to(device)
        local.to(device)
        self.model = Composite(glob=glob, local=local)
        self.model.to(device)
        self.epochs = epochs
        self.train_targets = []
        self.train_predictions = []
        self.learning_rate = learning_rate

    def load_params(self, w_glob, w_loc):
        if w_glob is not None:

            self.model.get_submodule("glob").load_state_dict(w_glob)
        if w_loc is not None:
            s_dict = {}
            for k in self.model.get_submodule("local").get_submodule("layer_list").state_dict():
                s_dict[k] = w_loc[k]
            self.model.get_submodule("local").get_submodule(
                "layer_list").load_state_dict(s_dict)
            print("loaded parameters")

    def train(self, dataloader):
        self.model.train()

        if self.optim == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(
            ), lr=self.learning_rate, momentum=self.momentum)
        elif self.optim == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(
            ), lr=self.learning_rate)
        else:
            raise ValueError

        loss_fun = nn.CrossEntropyLoss()

        num_batch = len(dataloader)

        performance = np.zeros(self.epochs*num_batch)
        for i in range(self.epochs):
            batch_targets = []
            batch_pred = []
            for batch, data in enumerate(dataloader):

                X, y = data[0].to(self.device), data[1].to(self.device)
                if torch.isnan(X).any():
                    print("Here")
                optimizer.zero_grad()
                pred = self.model(X)
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
                if torch.isnan(pred).any():
                    print("Here")
                try:
                    loss = loss_fun(pred, y)
                except Exception:
                    print("Here")

                loss.backward()
                optimizer.step()

                batch_targets = y
                batch_pred = pred.argmax(1)

                performance[i*num_batch+batch] = loss.item()

        return self.model.get_submodule("glob").state_dict(), self.model.get_submodule("local").get_submodule("layer_list").state_dict(), performance

    def test(self, dataloader):
        model = self.model

        model.eval()
        loss_fun = nn.CrossEntropyLoss()

        num_batch = len(dataloader)
        # num_batches = len(data)
        # generic train loop
        # epoch_loss = []
        performance = np.zeros((5, num_batch))
        with torch.no_grad():
            batch_targets = []
            batch_pred = []
            for batch, data in enumerate(dataloader):

                X, y = data[0].to(self.device), data[1].to(self.device)
                pred = model(X)
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
                loss = loss_fun(pred, y)

                batch_targets = y
                batch_pred = pred.argmax(1)

                performance[0, batch] = accuracy_score(
                    batch_targets.cpu(), batch_pred.cpu())
                performance[1, batch] = recall_score(
                    batch_targets.cpu(), batch_pred.cpu(), average='micro')
                performance[2, batch] = precision_score(
                    batch_targets.cpu(), batch_pred.cpu(), average='micro')
                performance[3, batch] = f1_score(
                    batch_targets.cpu(), batch_pred.cpu(), average='micro')
                performance[4, batch] = loss.item()

        return performance

    def __repr__(self):
        return self.model.__repr__()

import matplotlib.pyplot as plt
from models import Composite
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.classification import F1Score, Accuracy


class Client:
    def __init__(
        self,
        glob: torch.nn.Module,
        local: torch.nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int,
        learning_rate: float,
        optimizer,
        device,
        momentum=0,
    ):
        np.random.seed(0)
        torch.manual_seed(0)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.momentum = momentum
        self.optim = optimizer
        glob.to(device)
        local.to(device)
        self.glob = glob
        self.loc = local
        self.model = Composite(glob=glob, local=local)
        self.model.to(device)
        self.epochs = epochs
        self.train_targets = []
        self.train_predictions = []
        self.learning_rate = learning_rate
        self.transient_dim = local.output_dim
        self.metrics = F1Score("multiclass", num_classes=12).to(device)
        self.accuracy = Accuracy(task="multiclass", num_classes=12).to(device)

        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
            self.lopt = torch.optim.SGD(
                self.loc.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
            self.gopt = torch.optim.SGD(
                self.glob.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
        elif self.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate)
            self.lopt = torch.optim.Adam(
                self.loc.parameters(), lr=self.learning_rate)
            self.gopt = torch.optim.Adam(
                self.glob.parameters(), lr=self.learning_rate)
        else:
            raise ValueError

    def load_params(self, w_glob, w_loc):
        if w_glob is not None:
            self.model.get_submodule("glob").load_state_dict(w_glob)
        if w_loc is not None:
            s_dict = {}

            for k in (
                self.model.get_submodule("local")
                .get_submodule("layer_list")
                .state_dict()
            ):
                s_dict[k] = w_loc[k]
            self.model.get_submodule("local").get_submodule(
                "layer_list"
            ).load_state_dict(s_dict)

    def get_params(self):
        return (
            self.model.get_submodule("glob").state_dict(),
            self.model.get_submodule("local").get_submodule(
                "layer_list").state_dict(),
        )

    def train(self, t, r):
        np.random.seed(0)
        torch.manual_seed(0)
        self.model.train()

        num_batch = len(self.trainloader)

        performance = 0
        for i in range(self.epochs):
            batch_targets = []
            batch_pred = []
            for batch, data in enumerate(self.trainloader):
                X, y = data[0], data[1]
                if t == 0:
                    self.optimizer.zero_grad()
                else:
                    self.lopt.zero_grad()
                    self.gopt.zero_grad()

                pred, vec = self.model(X)
                loss = F.cross_entropy(pred, y)
                loss.backward()
                if t == 0:
                    self.optimizer.step()
                elif t == 1:
                    if r % 2 == 0:
                        self.lopt.step()
                    else:
                        self.gopt.step()
                else:
                    self.lopt.step()
                    self.gopt.step()

                performance += loss.item()

        return (
            self.model.get_submodule("glob").state_dict(),
            self.model.get_submodule("local").get_submodule(
                "layer_list").state_dict(),
            performance / (len(self.trainloader)*self.epochs),
        )

    def test(self):
        model = self.model

        model.eval()

        num_batch = len(self.testloader)

        f1 = np.zeros(num_batch)
        acc = np.zeros(num_batch)
        loss = np.zeros(num_batch)
        vec = np.zeros((num_batch, self.transient_dim))

        with torch.no_grad():
            for batch, data in enumerate(self.testloader):
                X, y = data[0], data[1]
                pred, features = model(X)
                f1[batch] = self.metrics(pred, y).item()
                acc[batch] = self.accuracy(pred, y).item()
                loss[batch] = F.cross_entropy(pred, y).item()
                vec[batch, :] = features.cpu().squeeze()[0]

        return f1.mean(), f1.std(), acc.mean(), acc.std(), vec

    def __repr__(self):
        return self.model.__repr__()

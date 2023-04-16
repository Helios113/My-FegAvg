# from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from models import Composite
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

np.random.seed(0)
torch.manual_seed(0)


class Client:
    def __init__(
        self,
        glob: torch.nn.Module,
        local: torch.nn.Module,
        dataloader: DataLoader,
        epochs: int,
        learning_rate: float,
        optimizer,
        device,
        momentum=0,
    ):
        self.dataloader = dataloader
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
            for k in (
                self.model.get_submodule("local")
                .get_submodule("layer_list")
                .state_dict()
            ):
                s_dict[k] = w_loc[k]
            self.model.get_submodule("local").get_submodule(
                "layer_list"
            ).load_state_dict(s_dict)
            print("loaded parameters")

    def train(self):
        self.model.train()

        if self.optim == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
        elif self.optim == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError

        num_batch = len(self.dataloader)

        performance = np.zeros(self.epochs * num_batch)
        for i in range(self.epochs):
            batch_targets = []
            batch_pred = []
            for batch, data in enumerate(self.dataloader):
                X, y = data[0], data[1]
                optimizer.zero_grad()
                pred = self.model(X)
                y = y.to("mps")
                loss = F.cross_entropy(pred, y)
                loss.backward()
                optimizer.step()

                batch_targets = y
                batch_pred = pred.argmax(1)

                performance[i * num_batch + batch] = loss.item()

        return (
            self.model.get_submodule("glob").state_dict(),
            self.model.get_submodule("local").get_submodule("layer_list").state_dict(),
            performance,
        )

    def test(self, dataloader):
        model = self.model

        model.eval()
        loss_fun = torch.nn.CrossEntropyLoss()

        num_batch = len(dataloader)
        # num_batches = len(data)
        # generic train loop
        # epoch_loss = []
        # performance = np.zeros((5, num_batch))
        loss = 0
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

                # performance[0, batch] = accuracy_score(
                #     batch_targets.cpu(), batch_pred.cpu()
                # )
                # performance[1, batch] = recall_score(
                #     batch_targets.cpu(), batch_pred.cpu(), average="micro"
                # )
                # performance[2, batch] = precision_score(
                #     batch_targets.cpu(), batch_pred.cpu(), average="micro"
                # )
                # performance[3, batch] = f1_score(
                #     batch_targets.cpu(), batch_pred.cpu(), average="micro"
                # )
                loss += loss.item()

        return loss / (batch + 1)

    def __repr__(self):
        return self.model.__repr__()

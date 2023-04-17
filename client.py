import matplotlib.pyplot as plt
from models import Composite
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.classification import F1Score

np.random.seed(0)
torch.manual_seed(0)


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
        self.trainloader = trainloader
        self.testloader = testloader
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
        
        
        
        self.metrics = F1Score("multiclass",num_classes=13).to(device)

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

        num_batch = len(self.trainloader)

        performance = 0
        for i in range(self.epochs):
            batch_targets = []
            batch_pred = []
            for batch, data in enumerate(self.trainloader):
                X, y = data[0], data[1]
                optimizer.zero_grad()
                pred = self.model(X)
                loss = F.cross_entropy(pred, y)
                loss.backward()
                optimizer.step()

                performance += loss.item()

        return (
            self.model.get_submodule("glob").state_dict(),
            self.model.get_submodule("local").get_submodule("layer_list").state_dict(),
            performance/len(self.trainloader),
        )

    def test(self):
        model = self.model

        model.eval()

        num_batch = len(self.testloader)
     
        loss = np.zeros(num_batch)
        with torch.no_grad():
            for batch, data in enumerate(self.testloader):
                X, y = data[0], data[1]
                pred = model(X)
                print(pred.shape)
                print(y.shape)
                loss[batch] = self.metrics(pred,y).item()
        return loss.mean(), loss.std()

    def __repr__(self):
        return self.model.__repr__()

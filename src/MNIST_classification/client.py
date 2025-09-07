import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from configs.mnist_config import *
from src.MNIST_classification.model import *

# normalization and tensor transformation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
)


def partition_dataset(dataset, client_id, num_clients):
    """
    Partition the dataset for a specific client.
    """
    # Get all indices
    indices = np.arange(len(dataset))

    # Split indices for this client
    split_size = len(dataset) // num_clients
    start = client_id * split_size
    end = (client_id + 1) * split_size if client_id != num_clients - 1 else len(dataset)
    client_indices = indices[start:end]
    return torch.utils.data.Subset(dataset, client_indices)


# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader):
        self.model = SimpleNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config=None):  # FIXED: Added 'config' argument
        return [param.data.numpy() for param in self.model.parameters()]

    def fit(self, parameters, config):
        """
        Update the model with the provided parameters and train on the local dataset.
        """
        idx = 0
        for param in self.model.parameters():
            param.data = torch.tensor(parameters[idx])
            idx += 1

        self.model.train()
        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local test dataset and return the accuracy.
        """
        idx = 0
        for param in self.model.parameters():
            param.data = torch.tensor(parameters[idx])
            idx += 1

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        return float(accuracy), len(self.test_loader.dataset), {"accuracy": accuracy}

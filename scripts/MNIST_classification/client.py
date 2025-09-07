import sys

import flwr as fl
from torch.utils.data import DataLoader
from torchvision import datasets

from configs.mnist_config import *
from src.MNIST_classification.client import *


def main():
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=DATASET_PATH, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATASET_PATH, train=False, download=True, transform=transform
    )

    # Get client ID and number of clients from command line arguments
    client_id = int(sys.argv[1])
    num_clients = int(sys.argv[2])

    # Partition the datasets
    train_subset = partition_dataset(train_dataset, client_id, num_clients)
    test_subset = partition_dataset(test_dataset, client_id, num_clients)

    # Use these in your DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BS, shuffle=True)

    # Start Flower client using the new recommended method
    fl.client.start_client(
        server_address="localhost:5000",
        client=FlowerClient(train_loader=train_loader, test_loader=test_loader),
    )


if __name__ == "__main__":
    main()

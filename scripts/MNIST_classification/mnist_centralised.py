import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from configs.mnist_config import *
from src.MNIST_classification.mnist_centralised import *
from src.MNIST_classification.model import *


def main():
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=DATASET_PATH, train=True, download=True, transform=normalise_transform()
    )
    test_dataset = datasets.MNIST(
        root=DATASET_PATH, train=False, download=True, transform=normalise_transform()
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            int(train_val_split * len(train_dataset)),
            len(train_dataset) - int(train_val_split * len(train_dataset)),
        ],
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)

    # Run training and testing
    train(model, train_loader, val_loader, criterion, optimizer, epochs=5)
    evaluate(model, test_loader, criterion)


if __name__ == "__main__":
    main()

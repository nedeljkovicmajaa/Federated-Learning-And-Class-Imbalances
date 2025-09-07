import sys

import flwr as fl

from configs.mnist_config import *
from src.MNIST_classification.server import *


def main():
    # Create strategy and run server
    strategy = SaveModelStrategy()

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="localhost:5000",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

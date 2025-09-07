import sys

import flwr as fl

from configs.fedavg_config import *
from src.seg_fedavg.client import UNetClient
from src.seg_fedavg.prepare_data import *


def main():
    # Parse arguments
    client_id = int(sys.argv[1])
    num_clients = int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else NUM_EPOCHS

    # Load data
    train_images, train_masks, val_images, val_masks, _, _ = data_loading()

    # Split training data for this client
    x_train, y_train = split_data(train_images, train_masks, num_clients, client_id)
    val_images, val_masks = split_data(val_images, val_masks, num_clients, client_id)

    # delete the original data to save memory
    del train_images, train_masks

    # Start Flower client
    client = UNetClient(
        x_train, y_train, val_images, val_masks, client_id=client_id, num_epochs=epochs
    )
    fl.client.start_client(
        server_address=hostname + ":8080",
        client=client.to_client(),  # Assuming your `client` is a subclass of flwr.client.NumPyClient
    )


if __name__ == "__main__":
    main()

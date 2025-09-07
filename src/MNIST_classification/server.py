import flwr as fl
import numpy as np

from configs.mnist_config import *
from src.MNIST_classification.model import *


# Define a custom strategy to save aggregated weights
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures, do_save=False):
        """
        Aggregate weights from clients and optionally save them to disk.
        Args:
            rnd (int): Current round number.
            results (list): List of tuples (client_id, weights, num_examples).
            failures (list): List of client IDs that failed to respond.
            do_save (bool): If True, save the aggregated weights to disk.
        Returns:
            aggregated_weights (list): Aggregated weights from clients.
        """

        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if do_save:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

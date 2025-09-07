import time

import flwr as fl
from flwr.common import parameters_to_ndarrays

from configs.fedavg_config import *
from src.seg_fedavg.custom_strategy import SaveModelStrategy
from src.seg_fedavg.model import UNetModel, dice_coef, dice_coef_loss, iou


def format_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def main():
    # Start timing when server begins
    start_time = time.time()

    # Initialize your strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,  # Require all clients to train
        min_evaluate_clients=NUM_CLIENTS,  # Require all clients to evaluate
        min_available_clients=NUM_CLIENTS,  # Require all clients to be available
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # After training is done, save the global model
    if strategy.final_parameters:
        # Convert to weights
        weights = parameters_to_ndarrays(strategy.final_parameters)

        # Load model and set weights
        model = UNetModel().model
        model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef, iou])
        model.set_weights(weights)

        # Save to disk
        model.save(INITIAL_FED_PATH + PROBLEM_TYPE + "/global_model_initial.h5")

    # End timing and print results
    end_time = time.time()
    total_duration = end_time - start_time
    formatted_time = format_seconds(total_duration)
    formatted_avg = format_seconds(total_duration / NUM_ROUNDS)

    # Prepare the log string
    log_lines = [
        "\nFederated Learning Completed",
        f"Total clients: {NUM_CLIENTS}",
        f"Rounds completed: {NUM_ROUNDS}",
        f"Total time: {total_duration:.2f} seconds",
        f"Total time (formated): {formatted_time}",
        f"Average per round: {total_duration/NUM_ROUNDS:.2f} seconds",
        f"Average per round (formated): {formatted_avg}",
    ]

    # Print to console
    for line in log_lines:
        print(line)

    # Write to file
    with open(INITIAL_FED_PATH + PROBLEM_TYPE + "timing_and_config.txt", "a") as f:
        for line in log_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()

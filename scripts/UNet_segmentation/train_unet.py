import warnings

import numpy as np

warnings.filterwarnings("ignore")

import time

import tensorflow as tf

from configs.unet_config import *
from src.UNet_segmentation.prepare_data import *
from src.UNet_segmentation.UNet_model import *


def main():
    # Load your dataset here
    train_images, train_masks, val_images, val_masks, test_images, test_masks = (
        load_data()
    )

    if SUBSET:
        train_images, train_masks = split_data(
            train_images, train_masks, NUM_CLIENTS, percent, PROBLEM_TYPE, CLIENT_ID
        )
        val_images, val_masks = split_data(
            val_images, val_masks, NUM_CLIENTS, percent, PROBLEM_TYPE, CLIENT_ID
        )

    if AUG:
        # rotate images for 90, 180, 270 degrees and append to the dataset
        train_images, train_masks = augment_data(train_images, train_masks)

    # shuffle corresponding images and masks
    train_images, train_masks = shuffle_data(train_images, train_masks)
    val_images, val_masks = shuffle_data(val_images, val_masks)
    test_images, test_masks = shuffle_data(test_images, test_masks)

    time_start = time.time()

    # Initialize and train the model
    unet = UNetModel()
    unet.compile()
    unet.train(
        train_images, train_masks, val_images, val_masks, batch_size=BS, epochs=EPOCHS
    )

    # Evaluate the model
    metrics = unet.evaluate(test_images, test_masks)
    print("Evaluation Metrics:", metrics)

    # Save the model, training history, and sample predictions
    unet.save_samples(test_images, test_masks)
    unet.save_model(MODEL_PATH)

    history = unet.history.history
    with open(HISTORY_TXT_PATH, "w") as f:
        for key in history.keys():
            f.write(f"{key}: {history[key]}\n")

    # Save model time and configuration
    time_end = time.time()
    total_duration = time_end - time_start
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_duration))
    formatted_avg = time.strftime("%H:%M:%S", time.gmtime(total_duration / EPOCHS))
    log_lines = [
        "\nUNet Segmentation Training Completed",
        f"Total time: {total_duration:.2f} seconds",
        f"Total time (formatted): {formatted_time}",
        f"Average per epoch: {total_duration / EPOCHS:.2f} seconds",
        f"Average per epoch (formatted): {formatted_avg}",
    ]
    print("\n".join(log_lines))
    with open(CONF_PATH, "a") as f:
        for line in log_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()

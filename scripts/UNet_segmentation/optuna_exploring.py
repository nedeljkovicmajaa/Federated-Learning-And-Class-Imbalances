from functools import partial

import numpy as np
import optuna

from configs.unet_config import *
from src.UNet_segmentation.metrics_utils import *
from src.UNet_segmentation.optuna import *
from src.UNet_segmentation.prepare_data import *


def main():
    # Load data
    train_images, train_masks, val_images, val_masks, test_images, test_masks = (
        load_data()
    )

    # shuffle corresponding images and masks
    train_images, train_masks = shuffle_data(train_images, train_masks)
    val_images, val_masks = shuffle_data(val_images, val_masks)
    test_images, test_masks = shuffle_data(test_images, test_masks)

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    pepared_objective = partial(
        objective,
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        test_images=test_images,
        test_masks=test_masks,
    )

    study.optimize(pepared_objective, n_trials=1)

    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    main()

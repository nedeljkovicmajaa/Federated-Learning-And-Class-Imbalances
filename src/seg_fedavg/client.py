import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from configs.fedavg_config import *
from src.seg_fedavg.model import UNetModel, dice_coef, dice_coef_loss, iou


class UNetClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, client_id=0, num_epochs=3):
        self.client_id = client_id  # Current client ID
        self.model_wrapper = UNetModel()  # Initialize the UNet model wrapper
        self.model = self.model_wrapper.model  # Get the underlying Keras model
        self.model_wrapper.compile()  # Compile the model with custom loss and metrics
        self.x_train, self.y_train = x_train, y_train  # Training data
        self.x_val, self.y_val = x_val, y_val  # Validation data
        self.num_epochs = num_epochs  # Number of epochs for training

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config=None):
        """
        Train the model on the local dataset.
        """
        self.model.set_weights(parameters)

        with open(
            INITIAL_FED_PATH + PROBLEM_TYPE + f"training_local{self.client_id}.txt", "a"
        ) as f:
            for epoch in range(self.num_epochs):

                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BS,
                    verbose=1,
                )
                loss = history.history["loss"][0]
                dice = history.history.get("dice_coef", [None])[0]
                iou = history.history.get("iou", [None])[0]
                f.write(
                    f"Training - Epoch {epoch+1}, Loss: {loss}, Dice: {dice}, IoU: {iou}\n"
                )

                val_loss, duz, val_metrics = self.evaluate(
                    parameters=self.model.get_weights()
                )
                val_dice = val_metrics.get("dice", None)
                val_iou = val_metrics.get("iou", None)
                f.write(
                    f"Validation - Epoch {epoch+1}, Loss: {val_loss}, Dice: {val_dice}, IoU: {val_iou}\n"
                )

            f.write("-----------------------------------------\n")
        self.model.save(
            INITIAL_FED_PATH
            + PROBLEM_TYPE
            + f"client_model_{self.client_id}_initial.h5"
        )

        return self.model.get_weights(), len(self.x_train), {}

    def predict(self, test_images):
        predictions = self.model.predict(test_images)
        return (predictions > 0.5).astype(int)

    def evaluate(self, parameters=None, config=None):
        """
        Evaluate the model on the validation dataset.
        """
        if parameters is not None:
            self.model.set_weights(parameters)

        predicted = self.predict(self.x_val)
        self.y_val = self.y_val.astype(int)

        metrics = {
            "accuracy": accuracy_score(self.y_val.flatten(), predicted.flatten()),
            "precision": precision_score(self.y_val.flatten(), predicted.flatten()),
            "recall": recall_score(self.y_val.flatten(), predicted.flatten()),
            "f1": f1_score(self.y_val.flatten(), predicted.flatten()),
            "dice": 2
            * np.sum(predicted * self.y_val)
            / (np.sum(predicted) + np.sum(self.y_val)),
            "iou": np.sum(predicted * self.y_val)
            / np.sum((predicted + self.y_val) > 0),
        }

        loss = 1 - metrics["dice"]

        dice = metrics["dice"]
        iou = metrics["iou"]
        print(f"Loss: {loss}, Dice: {dice}, IoU: {iou}")

        return loss, len(self.x_val), {"dice": float(dice), "iou": float(iou)}

    def save_model(self, path="unet_model.h5"):
        self.model.save(path)

    def load_model(self, path="unet_model.h5"):
        self.model = UNetModel()
        self.model.load_weights(path)

    def save_samples(self, test_images, test_masks):
        predicted = self.predict(test_images)

        three_samples = np.random.choice(range(len(test_images)), 3)

        plt.figure(figsize=(15, 5))
        plt.subplot(2, 3, 1)
        plt.imshow(test_masks[three_samples[0]].reshape(128, 128), cmap="gray")
        plt.title("Original Mask")
        plt.subplot(2, 3, 2)
        plt.imshow(test_masks[three_samples[1]].reshape(128, 128), cmap="gray")
        plt.title("Original Mask")
        plt.subplot(2, 3, 3)
        plt.imshow(test_masks[three_samples[2]].reshape(128, 128), cmap="gray")
        plt.title("Original Mask")
        plt.subplot(2, 3, 4)
        plt.imshow(predicted[three_samples[0]].reshape(128, 128), cmap="gray")
        plt.title("Predicted Mask")
        plt.subplot(2, 3, 5)
        plt.imshow(predicted[three_samples[1]].reshape(128, 128), cmap="gray")
        plt.title("Predicted Mask")
        plt.subplot(2, 3, 6)
        plt.imshow(predicted[three_samples[2]].reshape(128, 128), cmap="gray")
        plt.title("Predicted Mask")
        plt.savefig(SAMPLES_PATH)

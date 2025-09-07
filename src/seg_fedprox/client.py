import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from configs.fedprox_config import *
from src.seg_fedprox.model import UNetModel, dice_coef, dice_coef_loss, iou


class UNetClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, client_id=0, num_epochs=3):
        self.client_id = client_id  # Unique identifier for the client
        self.model_wrapper = UNetModel()  # Initialize the UNet model wrapper
        self.model = self.model_wrapper.model  # Access the underlying Keras model
        self.model_wrapper.compile()  # Compile the model with custom loss and metrics
        self.x_train, self.y_train = x_train, y_train  # Training data for the client
        self.x_val, self.y_val = x_val, y_val  # Validation data for the client
        self.num_epochs = num_epochs  # Number of epochs for training

        self.global_weights = None  # Track global parameters

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config=None):
        self.model.set_weights(parameters)
        self.global_weights = parameters  # Store global weights for proximal term

        # Custom training loop with proximal term
        optimizer = tf.keras.optimizers.Adam()
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        ).batch(BS)

        with open(
            INITIAL_FED_PATH + PROBLEM_TYPE + f"training_local{self.client_id}.txt", "a"
        ) as f:

            for epoch in range(self.num_epochs):
                for batch_x, batch_y in train_dataset:
                    with tf.GradientTape() as tape:
                        # Standard loss
                        pred = self.model(batch_x)
                        loss = dice_coef_loss(batch_y, pred)

                        # Proximal term calculation
                        proximal_term = 0.0
                        if self.global_weights is not None:
                            for local_w, global_w in zip(
                                self.model.get_weights(), self.global_weights
                            ):
                                proximal_term += tf.norm(local_w - global_w) ** 2

                        # Combined loss
                        total_loss = loss + (MU / 2) * proximal_term

                    gradients = tape.gradient(
                        total_loss, self.model.trainable_variables
                    )

                    optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )

                predicted = self.predict(self.x_train)
                self.y_train = self.y_train.astype(np.float32)
                dice = (
                    2
                    * np.sum(predicted * self.y_train)
                    / (np.sum(predicted) + np.sum(self.y_train))
                )
                iou_score = np.sum(predicted * self.y_train) / np.sum(
                    (predicted + self.y_train) > 0
                )
                f.write(
                    f"Training - Epoch {epoch+1}, Loss: {total_loss:.4f}, Dice: {dice:.4f}, IoU: {iou_score:.4f}\n"
                )

                val_loss, duz, val_metrics = self.evaluate(
                    parameters=self.model.get_weights()
                )
                val_dice = val_metrics.get("dice", None)
                val_iou = val_metrics.get("iou", None)
                f.write(
                    f"Validation - Epoch {epoch+1}, Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}\n"
                )
            f.write("-----------------------------------------\n")

        # Save the model after training
        self.model.save(
            INITIAL_FED_PATH + PROBLEM_TYPE + f"client_model_{self.client_id}.h5"
        )

        return self.model.get_weights(), len(self.x_train), {}

    def predict(self, test_images):
        predictions = self.model.predict(test_images)
        return (predictions > 0.8).astype(int)

    def evaluate(self, parameters=None, config=None):
        if parameters is not None:
            self.model.set_weights(parameters)

        predicted = self.predict(self.x_val)
        self.y_val = self.y_val.astype(int)

        metrics = {
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

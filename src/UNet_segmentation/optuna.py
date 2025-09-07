from functools import partial

import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping

from src.UNet_segmentation.metrics_utils import *


class UNetModel:
    def __init__(
        self,
        input_shape=(128, 128, 1),
        num_filters=8,
        dropout_rate=0.4,
        do_batch_norm=True,
        num_layers=3,
    ):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.do_batch_norm = do_batch_norm
        self.num_layers = num_layers
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.layers.Input(self.input_shape)

        x = inputs
        skip_connections = []

        for i in range(self.num_layers):
            x, skip = self.encoder_block(x, self.num_filters * (2**i))
            skip_connections.append(skip)

        x = self.conv2d_block(x, self.num_filters * (2**self.num_layers))

        for i in range(self.num_layers - 1, -1, -1):
            x = self.decoder_block(x, skip_connections[i], self.num_filters * (2**i))

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def encoder_block(self, input_tensor, num_filters):
        x = tf.keras.layers.Conv2D(
            num_filters, (3, 3), padding="same", activation="relu"
        )(input_tensor)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            num_filters, (3, 3), padding="same", activation="relu"
        )(x)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        skip = x
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        return x, skip

    def decoder_block(self, input_tensor, skip_tensor, num_filters):
        x = tf.keras.layers.Conv2DTranspose(
            num_filters, (3, 3), strides=2, padding="same"
        )(input_tensor)

        if x.shape[1] != skip_tensor.shape[1]:
            x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

        x = tf.keras.layers.concatenate([x, skip_tensor], axis=-1)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = self.conv2d_block(x, num_filters)
        return x

    def conv2d_block(self, input_tensor, num_filters):
        x = tf.keras.layers.Conv2D(
            num_filters, (3, 3), padding="same", activation="relu"
        )(input_tensor)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            num_filters, (3, 3), padding="same", activation="relu"
        )(x)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        return x

    def train(
        self, train_images, train_masks, val_images, val_masks, batch_size, epochs
    ):
        self.model.fit(
            train_images,
            train_masks,
            validation_data=(val_images, val_masks),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=3, monitor="val_loss")],
        )

    def evaluate(self, test_images, test_masks):
        return self.model.evaluate(test_images, test_masks)

    def save_model(self, file_path):
        self.model.save(file_path)

    def get_metrics(self, test_masks, predicted):
        metrics = {
            "accuracy": accuracy_score(test_masks.flatten(), predicted.flatten()),
            "precision": precision_score(test_masks.flatten(), predicted.flatten()),
            "recall": recall_score(test_masks.flatten(), predicted.flatten()),
            "f1": f1_score(test_masks.flatten(), predicted.flatten()),
            "dice": dice_coef(test_masks.flatten(), predicted.flatten()),
            "iou": iou(test_masks.flatten(), predicted.flatten()),
        }
        return metrics


def objective(
    trial, train_images, train_masks, val_images, val_masks, test_images, test_masks
):
    """
    Objective function for Optuna to optimize hyperparameters for the UNet model.
    """
    batch_size = trial.suggest_categorical("batch_size", [64])
    num_filters = trial.suggest_categorical("num_filters", [24])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.31])
    learning_rate = trial.suggest_categorical("learning_rate", [0.00985])
    num_layers = trial.suggest_categorical("num_layers", [3])

    model = UNetModel(
        input_shape=(128, 128, 1),
        num_filters=num_filters,
        dropout_rate=dropout_rate,
        num_layers=num_layers,
    )

    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[dice_coef_loss],
    )

    model.train(
        train_images,
        train_masks,
        val_images,
        val_masks,
        batch_size=batch_size,
        epochs=20,
    )

    # Evaluate model on test set
    metrics = model.evaluate(test_images, test_masks)
    f1 = metrics[1]  # F1-score is the second metric

    return f1

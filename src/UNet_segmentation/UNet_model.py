import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from configs.unet_config import *
from src.UNet_segmentation.metrics_utils import *


class UNetModel:
    def __init__(
        self,
        input_shape=(128, 128, 1),
        num_filters=num_filters,
        dropout_rate=dropout,
        do_batch_norm=batch_norm,
    ):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.do_batch_norm = do_batch_norm
        self.model = self.build_model()

    def conv2d_block(self, input_tensor, num_filters, kernel_size=3):
        x = tf.keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(LR),
        )(input_tensor)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(LR),
        )(x)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def encoder_block(self, input_tensor, num_filters):
        x = self.conv2d_block(input_tensor, num_filters)
        p = tf.keras.layers.MaxPooling2D((2, 2))(x)
        p = tf.keras.layers.Dropout(self.dropout_rate)(p)
        return x, p

    def decoder_block(self, input_tensor, skip_tensor, num_filters):
        x = tf.keras.layers.Conv2DTranspose(
            num_filters, (3, 3), strides=2, padding="same"
        )(input_tensor)
        x = tf.keras.layers.concatenate([x, skip_tensor])
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = self.conv2d_block(x, num_filters)
        return x

    def build_model(self):
        inputs = tf.keras.layers.Input(self.input_shape)

        # Encoder
        c1, p1 = self.encoder_block(inputs, self.num_filters)
        c2, p2 = self.encoder_block(p1, self.num_filters * 2)
        c3, p3 = self.encoder_block(p2, self.num_filters * 4)
        c4, p4 = self.encoder_block(p3, self.num_filters * 8)

        # Bottleneck
        c5 = self.conv2d_block(p4, self.num_filters * 16)

        # Decoder
        c6 = self.decoder_block(c5, c4, self.num_filters * 8)
        c7 = self.decoder_block(c6, c3, self.num_filters * 4)
        c8 = self.decoder_block(c7, c2, self.num_filters * 2)
        c9 = self.decoder_block(c8, c1, self.num_filters)

        # Output layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

        model = tf.keras.Model(inputs, outputs)

        return model

    def compile(self):
        self.model.compile(
            optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef, iou]
        )

    def train(
        self, train_images, train_masks, val_images, val_masks, batch_size=64, epochs=2
    ):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=patience, monitor="val_loss")
        ]
        if early_stopping == False:
            callbacks = None
        self.history = self.model.fit(
            train_images,
            train_masks,
            validation_data=(val_images, val_masks),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=True,
        )

    def predict(self, test_images):
        predictions = self.model.predict(test_images)
        return (predictions > 0.5).astype(int)

    def evaluate(self, test_images, test_masks):
        predicted = self.predict(test_images)
        test_masks = test_masks.astype(int)

        metrics = {
            "accuracy": accuracy_score(test_masks.flatten(), predicted.flatten()),
            "precision": precision_score(test_masks.flatten(), predicted.flatten()),
            "recall": recall_score(test_masks.flatten(), predicted.flatten()),
            "f1": f1_score(test_masks.flatten(), predicted.flatten()),
            "dice": 2
            * np.sum(predicted * test_masks)
            / (np.sum(predicted) + np.sum(test_masks)),
            "iou": np.sum(predicted * test_masks)
            / np.sum((predicted + test_masks) > 0),
        }
        return metrics

    def save_model(self, path="unet_model.h5"):
        self.model.save(path)

    def load_model(self, path="unet_model.h5"):
        self.model = self.build_model()
        self.model.load_weights(path)

    def plot_history(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["loss"], label="Loss")
        plt.plot(self.history.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["dice_coef"], label="Dice Coefficient")
        plt.plot(self.history.history["val_dice_coef"], label="Val Dice Coefficient")
        plt.legend()
        plt.title("Dice Coefficient")
        plt.savefig(HISTORY_PATH)

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

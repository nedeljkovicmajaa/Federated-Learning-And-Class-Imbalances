import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from configs.pretrained_config import *


def build_model(input_shape=(128, 128, 3)):
    """
    Build the FPN model with ResNet34 backbone.
    """
    model = sm.FPN(
        backbone_name="resnet34",
        encoder_weights="imagenet",
        input_shape=(128, 128, 3),
        classes=1,
        activation="sigmoid",
    )
    return model


def load_data():
    """
    Load training and validation data from .npy files.
    """
    train_images = np.load(base_path + "train_images.npy")
    train_masks = np.load(base_path + "train_masks.npy")
    val_images = np.load(base_path + "val_images.npy")
    val_masks = np.load(base_path + "val_masks.npy")
    return train_images, train_masks, val_images, val_masks


def create_augmenter():
    """
    Create an augmenter for data augmentation.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.5),
            tf.keras.layers.RandomZoom(0.5),
        ]
    )


def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the FPN model with ResNet34 backbone.
    """
    # Define preprocessor
    preprocess = sm.get_preprocessing("resnet34")

    def preprocess_fn(image, mask):
        image = tf.tile(image, [1, 1, 3])  # 1→3 channels
        image = preprocess(image)
        mask = tf.cast(mask, tf.float32)
        return image, mask

    # Prepare data for training and validation
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.map(preprocess_fn).batch(BS).prefetch(2)
    val_ds = val_ds.map(preprocess_fn).batch(BS).prefetch(2)

    # Compile the model with defined parameters and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
    )

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=patience),
        tf.keras.callbacks.ReduceLROnPlateau(),
    ]

    # Train initial layers
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=callbacks
    )

    # Unfreeze encoder for fine-tuning with smaller LR
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNING),  # Smaller LR for fine-tuning
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
    )

    # Fine-tune
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS_FINETUNING,
        callbacks=callbacks,
    )

    # Save the history log
    with open(HISTORY_TXT_PATH, "w") as f:
        for key, values in history.history.items():
            if key == "lr":
                continue
            f.write(f"{key}: {values}\n")

    return model


def evaluate_model(model, test_images, test_masks):
    """
    Evaluate trained model on the provided data.
    """
    # Ensure test images are in the correct format
    test_images = np.repeat(test_images, 3, axis=-1)
    # Apply preprocessing
    test_images = sm.get_preprocessing("resnet34")(test_images)

    # Predict masks
    preds = model.predict(test_images)
    binary_masks = (preds > 0.5).astype(np.float32)

    # Ensure test masks are in the correct format
    test_masks = test_masks.astype(np.float32)
    test_masks = np.clip(test_masks, 0, 1)

    # Calculate metrics
    iou = sm.metrics.iou_score(test_masks, binary_masks)
    dice = sm.metrics.f1_score(test_masks, binary_masks)
    print(f"IoU: {iou:.3f}, Dice: {dice:.3f}")

    # Plot a result sample from the test set
    idx = np.random.randint(len(test_images))
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Input")
    plt.imshow(test_images[idx][..., 0], cmap="gray")
    plt.subplot(132)
    plt.title("Prediction")
    plt.imshow(binary_masks[idx, ..., 0], cmap="gray")
    plt.subplot(133)
    plt.title("Ground Truth")
    plt.imshow(test_masks[idx, ..., 0], cmap="gray")
    plt.savefig(RESULTS_IMAGE_PATH)

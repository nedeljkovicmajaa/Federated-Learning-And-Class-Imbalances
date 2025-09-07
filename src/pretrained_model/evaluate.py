import os

import numpy as np
import tensorflow as tf

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from configs.pretrained_config import *
from src.pretrained_model.train import *


def load_trained_model(model_path):
    """
    Load the saved model with custom objects.
    """
    model = build_model()
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    return model


def load_and_preprocess_data():
    """
    Data loading and preprocessing.
    """
    return (
        np.load(base_path + "test_images.npy"),
        np.load(base_path + "test_masks.npy"),
    )


def evaluate_model(model, test_images, test_masks):
    """
    Evaluate the model and calculate metrics.
    """
    # Preprocess test data
    test_images = np.repeat(test_images, 3, axis=-1)  # 1→3 channels
    test_images = sm.get_preprocessing("resnet34")(test_images)

    # Predict and threshold
    preds = model.predict(test_images)
    binary_masks = (preds > 0.5).astype(np.float32)  # Key change

    # Calculate metrics on BINARY masks
    iou = sm.metrics.iou_score(test_masks, binary_masks)
    dice = sm.metrics.f1_score(test_masks, binary_masks)
    print(f"IoU: {iou:.3f}, Dice: {dice:.3f}")

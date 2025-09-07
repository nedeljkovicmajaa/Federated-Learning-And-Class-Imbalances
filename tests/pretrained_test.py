import os

import numpy as np
import pytest
import tensorflow as tf

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from src.pretrained_model.train import (build_model, create_augmenter,
                                        evaluate_model)


@pytest.fixture
def dummy_model():
    model = build_model()
    return model


@pytest.fixture
def dummy_input_batch():
    # Create dummy grayscale 128x128 batch (batch_size=4)
    images = np.random.rand(4, 128, 128, 1).astype(np.float32)
    masks = np.random.randint(0, 2, (4, 128, 128, 1)).astype(np.float32)
    return images, masks


def test_model_output_shape(dummy_model):
    x = tf.random.normal((2, 128, 128, 3))  # 3-channel input
    y = dummy_model(x)
    assert y.shape == (2, 128, 128, 1), "Model output should be (batch, H, W, 1)"


def test_model_compiles_with_losses_and_metrics(dummy_model):
    try:
        dummy_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=sm.losses.dice_loss,
            metrics=[
                sm.metrics.IOUScore(threshold=0.5),
                sm.metrics.FScore(threshold=0.5),
            ],
        )
    except Exception as e:
        pytest.fail(f"Model compilation failed: {e}")


def test_augmenter_runs():
    augmenter = create_augmenter()
    dummy_data = tf.random.normal((2, 128, 128, 3))
    try:
        augmented = augmenter(dummy_data)
        assert augmented.shape == (2, 128, 128, 3)
    except Exception as e:
        pytest.fail(f"Augmentation failed: {e}")


def test_model_can_predict(dummy_model, dummy_input_batch):
    dummy_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
    )
    x, _ = dummy_input_batch
    x_rgb = np.repeat(x, 3, axis=-1)  # Convert to 3-channel
    preds = dummy_model.predict(x_rgb)
    assert preds.shape == (
        4,
        128,
        128,
        1,
    ), "Prediction shape should be (batch_size, H, W, 1)"
    assert np.all(preds >= 0.0) and np.all(
        preds <= 1.0
    ), "Prediction data should be in [0, 1] range"

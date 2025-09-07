import numpy as np
import pytest
import tensorflow as tf
from src.seg_fedprox.client import UNetClient  # adjust path as needed
from src.seg_fedprox.model import UNetModel, dice_coef, dice_coef_loss
from src.seg_fedprox.prepare_data import shuffle_data, split_data

# Dummy input shape and batch for UNet
INPUT_SHAPE = (128, 128, 1)
BATCH_SIZE = 4


@pytest.fixture
def dummy_data():
    x = np.random.rand(BATCH_SIZE, *INPUT_SHAPE).astype(np.float32)
    y = np.random.randint(0, 2, (BATCH_SIZE, *INPUT_SHAPE)).astype(np.float32)
    return x, y


def test_model_output_shape(dummy_data):
    model = UNetModel().model
    output = model(dummy_data[0])
    assert output.shape == dummy_data[1].shape, "Output shape mismatch"


def test_compile_model():
    model_wrapper = UNetModel()
    model_wrapper.compile()
    model = model_wrapper.model
    assert model.optimizer is not None, "Model is not compiled properly"
    assert model.loss is not None, "Model is not compiled properly"


def test_dice_coef_range(dummy_data):
    y_true, y_pred = dummy_data
    score = dice_coef(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    assert 0.0 <= score <= 1.0, "Dice coefficient out of bounds"


def test_data_shuffle_preserves_shape(dummy_data):
    x, y = dummy_data
    x_shuf, y_shuf = shuffle_data(x, y)
    assert x.shape == x_shuf.shape, "Shuffled images shape does not match original"
    assert y.shape == y_shuf.shape, "Shuffled masks shape does not match original"


def test_data_splitting():
    x = np.random.rand(20, *INPUT_SHAPE)
    y = np.random.randint(0, 2, (20, *INPUT_SHAPE))
    num_clients = 4
    client_id = 1
    x_part, y_part = split_data(x, y, num_clients, client_id)
    assert (
        x_part.shape[1:] == INPUT_SHAPE
    ), "Split data shape does not match input shape"
    assert len(x_part) > 0, "Split data should not be empty"

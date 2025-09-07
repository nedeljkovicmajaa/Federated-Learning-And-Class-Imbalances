import numpy as np
import pytest
import tensorflow as tf
from src.UNet_segmentation.metrics_utils import (combined_loss, dice_coef,
                                                 dice_coef_loss, iou)
from src.UNet_segmentation.prepare_data import *
from src.UNet_segmentation.UNet_model import UNetModel  # Adjust import path


@pytest.fixture
def dummy_data():
    # Create dummy images and masks: 10 samples, 64x64 size, 1 channel
    images = np.arange(10 * 64 * 64).reshape(10, 64, 64).astype(np.float32)
    masks = np.arange(10 * 64 * 64).reshape(10, 64, 64).astype(np.float32)
    return images, masks


def test_basic_functionality():
    y_true = tf.constant([1, 0, 1, 1], dtype=tf.float32)
    y_pred = tf.constant([0.9, 0.1, 0.8, 0.6], dtype=tf.float32)

    dice = dice_coef(y_true, y_pred).numpy()
    assert 0 <= dice <= 1, "Dice coeficient is not calculated correctly"

    dice_loss = dice_coef_loss(y_true, y_pred).numpy()
    assert dice_loss == 1 - dice, "Dice loss is not calculated correctly"

    loss = combined_loss(y_true, y_pred).numpy()
    assert loss > 0, "Combined loss is not calculated correctly"

    iou_val = iou(y_true, y_pred).numpy()
    assert 0 <= iou_val <= 1, "IoU is not calculated correctly"


def test_augment_data_shapes(dummy_data):
    images, masks = dummy_data
    aug_images, aug_masks = augment_data(images, masks)
    # Original 10 + 3 rotations x 10 = 40 samples total
    assert (
        aug_images.shape[0] == 40
    ), "Augmented images do not have expected number of samples"
    assert (
        aug_masks.shape[0] == 40
    ), "Augmented masks do not have expected number of samples"
    assert (
        aug_images.shape[1:] == images.shape[1:]
    ), "Augmented images do not have expected shape"
    assert (
        aug_masks.shape[1:] == masks.shape[1:]
    ), "Augmented masks do not have expected shape"


def test_shuffle_data_alignment(dummy_data):
    images, masks = dummy_data
    shuffled_images, shuffled_masks = shuffle_data(images, masks)
    # Should have same shape after shuffle
    assert (
        shuffled_images.shape == images.shape
    ), "Shuffled images do not have expected shape"
    assert (
        shuffled_masks.shape == masks.shape
    ), "Shuffled masks do not have expected shape"
    # After shuffle, ordering changes but alignment kept
    # Check that masks[i] corresponds to images[i] by comparing their flattened sum
    for i in range(len(shuffled_images)):
        assert np.sum(shuffled_images[i]) == np.sum(
            shuffled_masks[i]
        ), "Shuffled images and masks are not aligned"


def test_split_data_statistical_het(dummy_data):
    images, masks = dummy_data
    num_clients = 4
    percent = 0.3
    problem_type = "statistical_het_example"

    # Test client 0
    imgs0, msks0 = split_data(images, masks, num_clients, percent, problem_type, 0)
    # Should have ~30% of data for client 0
    assert len(imgs0) == int(
        0.3 * len(images)
    ), "Client 0 does not have expected number of samples"

    # Test client 1
    imgs1, msks1 = split_data(images, masks, num_clients, percent, problem_type, 1)
    assert len(imgs1) == int(
        0.3 * len(images)
    ), "Client 1 does not have expected number of samples"

    # Test client 2 (other clients)
    imgs2, msks2 = split_data(images, masks, num_clients, percent, problem_type, 2)
    expected_len = (len(images) - int(0.3 * len(images))) // (num_clients - 1)
    assert (
        len(imgs2) == expected_len
    ), "Client 2 does not have expected number of samples"

    # Check alignment between images and masks for one client
    for i in range(len(imgs2)):
        assert np.sum(imgs2[i]) == np.sum(
            msks2[i]
        ), "Images and masks are not aligned for client 2"


def test_split_data_default(dummy_data):
    images, masks = dummy_data
    num_clients = 5
    percent = 0.2
    problem_type = "non_het"

    # Each client should get approx equal partition except last
    for client_id in range(num_clients):
        imgs, msks = split_data(
            images, masks, num_clients, percent, problem_type, client_id
        )
        if client_id < num_clients - 1:
            assert (
                len(imgs) == len(images) // num_clients
            ), "Client {} does not have expected number of samples".format(client_id)
        else:
            # Last client gets the remainder
            assert len(imgs) == len(images) - (num_clients - 1) * (
                len(images) // num_clients
            ), "Last client does not have expected number of samples"

        for i in range(len(imgs)):
            assert np.sum(imgs[i]) == np.sum(
                msks[i]
            ), "Images and masks are not aligned for client {}".format(client_id)


def test_model_compile(dummy_data):
    images, masks = dummy_data
    model = UNetModel()
    model.compile()
    assert model.model.loss is not None, "Model is not compiled correctly"

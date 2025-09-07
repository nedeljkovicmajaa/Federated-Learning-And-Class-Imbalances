import numpy as np

from configs.fedavg_config import *


def shuffle_data(images, masks):
    """
    Shuffle the images and masks in unison.
    """
    idx = np.random.permutation(len(images))
    images = images[idx]
    masks = masks[idx]
    return np.array(images), np.array(masks)


def data_loading():
    """
    Load the dataset, apply data augmentation if specified and shuffle the test data.
    """
    # Load your dataset here
    train_images, train_masks = np.load(DATA_PATH + "/train_images.npy"), np.load(
        DATA_PATH + "/train_masks.npy"
    )
    val_images, val_masks = np.load(DATA_PATH + "/val_images.npy"), np.load(
        DATA_PATH + "/val_masks.npy"
    )
    test_images, test_masks = np.load(DATA_PATH + "/test_images.npy"), np.load(
        DATA_PATH + "/test_masks.npy"
    )

    if AUG:
        # rotate images for 90, 180, 270 degrees and append to the dataset
        train_images_rotated90, train_masks_rotated90 = np.rot90(
            train_images, axes=(1, 2), k=1
        ), np.rot90(train_masks, axes=(1, 2), k=1)
        train_images_rotated180, train_masks_rotated180 = np.rot90(
            train_images, axes=(1, 2), k=2
        ), np.rot90(train_masks, axes=(1, 2), k=2)
        train_images_rotated270, train_masks_rotated270 = np.rot90(
            train_images, axes=(1, 2), k=3
        ), np.rot90(train_masks, axes=(1, 2), k=3)

        train_images = np.concatenate(
            (
                train_images,
                train_images_rotated90,
                train_images_rotated180,
                train_images_rotated270,
            ),
            axis=0,
        )
        train_masks = np.concatenate(
            (
                train_masks,
                train_masks_rotated90,
                train_masks_rotated180,
                train_masks_rotated270,
            ),
            axis=0,
        )

    # Shuffle the test data
    test_images, test_masks = shuffle_data(test_images, test_masks)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def split_data(images, masks, num_clients, client_id):
    """
    Split the dataset into `num_clients` and return the part for `client_id`.
    """
    if "statistical_het" in PROBLEM_TYPE:
        total_size = len(images)
        split_index = int(percent * total_size)

        if client_id == 0:
            images = images[:split_index]
            masks = masks[:split_index]
        elif client_id == 1:
            images = images[split_index : 2 * split_index]
            masks = masks[split_index : 2 * split_index]
        else:
            split_index = 2 * split_index
            num_clients = num_clients - 1  # Adjust for the first two clients
            client_id = client_id - 1

            len_data = (total_size - split_index) // (num_clients - 1)
            start_index = split_index + (client_id - 1) * len_data
            end_index = start_index + len_data
            images = images[start_index:end_index]
            masks = masks[start_index:end_index]

        images = np.array(images)
        masks = np.array(masks)
        print(f"Client {client_id} data size: {len(images)}")

        images, masks = shuffle_data(images, masks)
        print(f"Total size {total_size} data shuffled size: {len(images)}")
        return images, masks
    else:
        total_size = len(images)
        part_size = total_size // num_clients
        start = client_id * part_size
        end = (
            (client_id + 1) * part_size if client_id != num_clients - 1 else total_size
        )
        return images[start:end], masks[start:end]

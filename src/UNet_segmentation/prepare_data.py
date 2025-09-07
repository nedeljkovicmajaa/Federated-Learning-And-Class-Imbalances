import numpy as np

from configs.unet_config import *


def load_data():
    """
    Load the dataset from the specified path.
    """
    train_images, train_masks = np.load(DATA_PATH + "train_images.npy"), np.load(
        DATA_PATH + "train_masks.npy"
    )
    val_images, val_masks = np.load(DATA_PATH + "val_images.npy"), np.load(
        DATA_PATH + "val_masks.npy"
    )
    test_images, test_masks = np.load(DATA_PATH + "test_images.npy"), np.load(
        DATA_PATH + "test_masks.npy"
    )

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def augment_data(images, masks):
    """
    Augment the dataset by rotating images and masks by 90, 180, and 270 degrees.
    """
    images_rotated90 = np.rot90(images, axes=(1, 2), k=1)
    masks_rotated90 = np.rot90(masks, axes=(1, 2), k=1)
    images_rotated180 = np.rot90(images, axes=(1, 2), k=2)
    masks_rotated180 = np.rot90(masks, axes=(1, 2), k=2)
    images_rotated270 = np.rot90(images, axes=(1, 2), k=3)
    masks_rotated270 = np.rot90(masks, axes=(1, 2), k=3)

    images = np.concatenate(
        (images, images_rotated90, images_rotated180, images_rotated270), axis=0
    )
    masks = np.concatenate(
        (masks, masks_rotated90, masks_rotated180, masks_rotated270), axis=0
    )

    return images, masks


def shuffle_data(images, masks):
    """
    Shuffle the images and masks in unison.
    """
    idx = np.random.permutation(len(images))
    images = images[idx]
    masks = masks[idx]
    return np.array(images), np.array(masks)


def split_data(images, masks, num_clients, percent, problem_type, client_id):
    """
    Split the dataset into `num_clients` and return the part for `client_id`.
    """
    if "statistical_het" in problem_type:
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

import numpy as np
import torch

from configs.unet_paper_config import *


def augmentation(images, masks):
    """
    Double the dataset by applying random augmentation methods per image.
    """
    # rotate 90, 180, 270, flip horizontal, flip vertical
    images90, masks90 = torch.rot90(images, 1, [2, 3]), torch.rot90(masks, 1, [1, 2])
    images180, masks180 = torch.rot90(images, 2, [2, 3]), torch.rot90(masks, 2, [1, 2])
    images270, masks270 = torch.rot90(images, 3, [2, 3]), torch.rot90(masks, 3, [1, 2])
    images_h, masks_h = torch.flip(images, [2]), torch.flip(masks, [1])
    images_v, masks_v = torch.flip(images, [3]), torch.flip(masks, [2])

    # random select one augmentation method per image
    images_transform, masks_transform = [], []
    for i in range(images.size(0)):
        idx = np.random.randint(0, 5)
        if idx == 0:
            images_transform.append(images90[i])
            masks_transform.append(masks90[i])
        elif idx == 1:
            images_transform.append(images180[i])
            masks_transform.append(masks180[i])
        elif idx == 2:
            images_transform.append(images270[i])
            masks_transform.append(masks270[i])
        elif idx == 3:
            images_transform.append(images_h[i])
            masks_transform.append(masks_h[i])
        elif idx == 4:
            images_transform.append(images_v[i])
            masks_transform.append(masks_v[i])

    # append transformed and original data
    images_transform = torch.stack(images_transform)
    masks_transform = torch.stack(masks_transform)
    images = torch.cat([images, images_transform], dim=0)
    masks = torch.cat([masks, masks_transform], dim=0)

    return images, masks


def shuffle_manual(images, masks):
    """
    Shuffle the dataset such that images and masks are still corresponding.
    """
    idx = torch.randperm(images.size(0))
    images = images[idx]
    masks = masks[idx]
    return images, masks


def prepare_data():
    """
    Prepare the training and validation dataloaders (load, dim adapt, augment, shuffle).
    """
    # load data
    train_images, train_masks = np.load(train_save), np.load(train_mask_save)
    val_images, val_masks = np.load(val_save), np.load(val_mask_save)

    if TESTING:
        train_images, train_masks = train_images[:10], train_masks[:10]
        val_images, val_masks = val_images[:10], val_masks[:10]

    # prepare data - dimension adaptation
    train_images, train_masks = np.transpose(train_images, (0, 3, 1, 2)), np.transpose(
        train_masks, (0, 3, 1, 2)
    )
    val_images, val_masks = np.transpose(val_images, (0, 3, 1, 2)), np.transpose(
        val_masks, (0, 3, 1, 2)
    )
    train_masks, val_masks = train_masks[:, 0, :, :], val_masks[:, 0, :, :]

    # data augmentation only for training set
    train_images, train_masks = augmentation(
        torch.tensor(train_images), torch.tensor(train_masks)
    )
    val_images, val_masks = torch.tensor(val_images), torch.tensor(val_masks)

    # shuffle the dataset
    train_images, train_masks = shuffle_manual(train_images, train_masks)
    val_images, val_masks = shuffle_manual(val_images, val_masks)

    # create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_images), torch.tensor(train_masks)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_images), torch.tensor(val_masks)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader

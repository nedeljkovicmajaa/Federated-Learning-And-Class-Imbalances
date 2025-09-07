import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """
    Converts the segmentation target into a one-hot encoded format.
    """
    dice_target = target.clone()  # Clone to avoid modifying the original tensor

    if ignore_index >= 0:
        # Create a mask to identify ignored pixels
        ignore_mask = torch.eq(
            target, ignore_index
        )  # Mask where target == ignore_index
        dice_target[ignore_mask] = 0  # Temporarily set ignored pixels to class 0

        # Convert to one-hot encoding
        dice_target = dice_target.long()  # Ensure integer values
        dice_target = nn.functional.one_hot(
            dice_target, num_classes
        ).float()  # Convert to one-hot

        # Restore ignored pixels in one-hot format
        dice_target[ignore_mask] = ignore_index
    else:
        # Convert directly to one-hot encoding if ignore_index is not used
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    # Change shape from [N, H, W, C] to [N, C, H, W] (required format for PyTorch models)
    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(
    x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6
):
    """
    Computes the Dice coefficient, averaged over a batch.
    """
    d = 0.0  # Initialize Dice score accumulator
    batch_size = x.shape[0]  # Get batch size

    for i in range(batch_size):
        # Flatten both prediction and ground truth for easy calculation
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)

        if ignore_index >= 0:
            # Create a mask to exclude ignored pixels
            roi_mask = torch.ne(t_i, ignore_index)  # Identify non-ignored pixels
            x_i = x_i[roi_mask]  # Apply mask to predictions
            t_i = t_i[roi_mask]  # Apply mask to ground truth

        # Compute intersection (dot product counts common 1s in both tensors)
        inter = torch.dot(x_i, t_i)

        # Compute sum of both sets (union)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)

        # Handle case where both sets are empty
        if sets_sum == 0:
            sets_sum = 2 * inter

        # Compute Dice coefficient for this batch item
        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    # Return the average Dice coefficient over the batch
    return d / batch_size


def multiclass_dice_coeff(
    x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6
):
    """
    Computes the average Dice coefficient across multiple classes.
    """
    dice = 0.0  # Initialize total Dice score

    # Loop through each class (channel) in the one-hot encoded mask
    for channel in range(x.shape[1]):
        # Compute Dice coefficient for each class separately
        dice += dice_coeff(
            x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon
        )

    # Return the average Dice coefficient across all classes
    return dice / x.shape[1]


def dice_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    multiclass: bool = False,
    ignore_index: int = -100,
):
    """
    Computes the Dice loss, which is 1 - Dice coefficient (to be minimized).
    """
    # Apply softmax to normalize predictions to probability values
    x = nn.functional.softmax(x, dim=1)

    # Use multiclass Dice coefficient if multiple classes, otherwise binary Dice coefficient
    fn = multiclass_dice_coeff if multiclass else dice_coeff

    # Compute Dice loss (1 - Dice score)
    return 1 - fn(x, target, ignore_index=ignore_index)

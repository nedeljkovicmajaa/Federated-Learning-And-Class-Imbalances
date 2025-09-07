import torch
from torch import nn

import src.UNet_paper.distributed_utils as utils
from src.UNet_paper.dice_coefficient_loss import build_target, dice_loss


def criterion(
    inputs,
    target,
    loss_weight=None,
    num_classes: int = 2,
    dice: bool = True,
    ignore_index: int = -100,
):
    """
    Compute the loss function.
    """
    losses = {}
    for name, x in inputs.items():
        target = target.long()
        loss = nn.functional.cross_entropy(
            x, target, weight=loss_weight, ignore_index=ignore_index
        )  # Compute cross-entropy loss
        if dice is True:
            dice_target = build_target(
                target, num_classes, ignore_index
            )  # Build the target for Dice loss
            loss += dice_loss(
                x, dice_target, multiclass=True, ignore_index=ignore_index
            )  # Add Dice loss

        losses[name] = loss  # Store the loss for each output

    # If there is only one output, return its loss
    if len(losses) == 1:
        return losses["out"]

    # Otherwise, return a combination of the losses (main output + auxiliary)
    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)  # Confusion matrix for evaluation
    dice = utils.DiceCoefficient(
        num_classes=num_classes, ignore_index=255
    )  # Dice coefficient metric
    metric_logger = utils.MetricLogger(delimiter="  ")  # Logger for metrics
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)  # Get model output
            output = output["out"]  # Extract the main output

            # Update the confusion matrix and Dice coefficient
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

    return confmat, dice.value.item()


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    num_classes,
    lr_scheduler,
    print_freq=1,
    scaler=None,
):
    """
    Train the model for one epoch.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")  # Logger for metrics
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )  # Learning rate logger
    header = "Epoch: [{}]".format(epoch)  # Epoch header for logging

    if num_classes == 2:
        loss_weight = torch.as_tensor(
            [1.0, 2.0], device=device
        )  # Weight for binary classification
    else:
        loss_weight = None  # No weight for multi-class if not specified

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(
            enabled=scaler is not None
        ):  # Use mixed precision if available
            output = model(image)
            loss = criterion(
                output, target, loss_weight, num_classes=num_classes, ignore_index=255
            )  # Calculate the loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            scaler.step(optimizer)  # Step the optimizer
            scaler.update()  # Update the scaler
        else:
            loss.backward()  # Standard backpropagation if no scaler
            optimizer.step()  # Step the optimizer

        lr_scheduler.step()  # Update the learning rate scheduler

        lr = optimizer.param_groups[0]["lr"]  # Get the current learning rate
        metric_logger.update(loss=loss.item(), lr=lr)  # Log the loss and learning rate

    # Return the average loss for the epoch and the learning rate
    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
):
    """
    A learning rate scheduler.
    """
    assert num_step > 0 and epochs > 0  # Ensure valid steps and epochs
    if warmup is False:
        warmup_epochs = 0  # If no warmup, set warmup_epochs to 0

    def f(x):
        """
        Return the learning rate scaling factor based on the number of steps.
        During warmup, the learning rate scales from warmup_factor to 1.
        After warmup, the learning rate decays from 1 to 0.
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)  # Warmup scaling factor
            return (
                warmup_factor * (1 - alpha) + alpha
            )  # Warmup phase: lr scales from warmup_factor to 1
        else:
            # Decay phase: lr scales from 1 to 0 after warmup
            return (
                1
                - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)
            ) ** 0.9

    # Return the learning rate scheduler using the scaling function
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

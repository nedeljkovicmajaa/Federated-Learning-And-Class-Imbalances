import datetime
import os
import time
import warnings

import numpy as np
import torch

from configs.unet_paper_config import *
from src.UNet_paper.prepare_data import *
from src.UNet_paper.train_and_eval import (create_lr_scheduler, evaluate,
                                           train_one_epoch)
from src.UNet_paper.unet import UNet

warnings.filterwarnings("ignore")


def train_unet():
    """
    Train the UNet model for tumor segmentation.
    """
    # segmentation tumor + background
    num_classes = init_num_classes + 1

    # prepare dataloaders
    train_loader, val_loader = prepare_data()

    # create model
    model = UNet(in_channels=1, num_classes=num_classes, base_c=base_num_filters)
    model.to(device)

    # create optimizer
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # create learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_loader), num_epochs, warmup=True
    )

    best_dice = 0.0  # track best dice coefficient
    start_time = time.time()  # track training time

    # train for num_epochs
    for epoch in range(num_epochs):
        # train one epoch + get loss and updated lr
        mean_loss, lr_new = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            num_classes,
            lr_scheduler=lr_scheduler,
            print_freq=print_freq,
        )
        # evaluate on validation set
        confmat, dice = evaluate(
            model, val_loader, device=device, num_classes=num_classes
        )

        # print and save results
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")

        with open(results_file, mode="a") as f:
            train_info = (
                f"[epoch: {epoch}]\n"
                f"train_loss: {mean_loss:.4f}\n"
                f"lr: {lr_new:.6f}\n"
                f"dice coefficient: {dice:.3f}\n"
            )
            f.write(train_info + val_info + "\n\n")

        # save best model
        if best_dice < dice:
            best_dice = dice
        else:
            continue

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }

        torch.save(save_file, save_weights_path)

    # print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == "__main__":
    train_unet()

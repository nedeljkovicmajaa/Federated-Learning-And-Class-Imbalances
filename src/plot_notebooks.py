import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def SHAP_analysis(shap_df, low_params, grouped_data):
    """
    Create a SHAP analysis plot with a barplot for SHAP values and dot plots for low-impact hyperparameters.
    Parameters:
    - shap_df: DataFrame containing SHAP values with columns 'Hyperparameter', 'Mean|SHAP|', and 'Std'.
    - low_params: DataFrame containing low-impact hyperparameters with a 'Hyperparameter' column
    - grouped_data: DataFrame containing grouped data for low-impact hyperparameters with a 'mean_f1' column.
    """
    fig = plt.figure(figsize=(16, 7))

    # reduce vertical spacing between right plots
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 4], wspace=0.15)
    gs_right = gs[0, 1].subgridspec(3, 1, hspace=0.32)  # smaller hspace (default 0.5)

    # Left barplot: make bars thinner (width < 0.8 default)
    ax0 = fig.add_subplot(gs[0, 0])
    sns.barplot(
        x="Mean |SHAP|",
        y="Hyperparameter",
        data=shap_df,
        palette="crest",
        ax=ax0,
        width=0.5,  # narrower bars
    )
    ax0.errorbar(
        shap_df["Mean |SHAP|"],
        np.arange(len(shap_df)),
        xerr=shap_df["Std"].to_numpy(),
        fmt="none",
        ecolor="black",
        capsize=4,
        elinewidth=1.5,
    )

    # Increase font sizes for axis labels and ticks on left plot
    ax0.set_title(
        "SHAP Importance of Hyperparameters", fontsize=16, fontweight="normal"
    )
    ax0.set_xlabel("Mean |SHAP|", fontsize=14, fontweight="normal")
    ax0.set_ylabel(None)
    wrapped_labels = [label.replace(" ", "\n") for label in shap_df["Hyperparameter"]]
    ax0.set_yticklabels(wrapped_labels, fontsize=13)
    ax0.tick_params(axis="both", labelsize=13)
    ax0.grid(axis="x", linestyle="--", alpha=0.6)
    sns.despine(ax=ax0, top=True, right=True)

    # Right dot plots: larger font, tighter spacing, slightly smaller dots
    for i, param in enumerate(low_params["Hyperparameter"].values):
        ax = fig.add_subplot(gs_right[i, 0])
        dfg = grouped_data[param]
        groups = dfg.index.astype(str)
        means = dfg["mean_f1"].values
        sorted_indices = np.argsort(groups.astype(int))[::-1]
        means = means[sorted_indices]
        groups = groups[sorted_indices]

        if i == 0:
            ax.set_title("Low-Impact Hyperparameters", fontsize=16, fontweight="normal")

        max_idx = np.argmax(means)

        # base dots smaller
        ax.scatter(
            means,
            range(len(groups)),
            color="#728F72FF",
            s=90,  # smaller marker size
            zorder=3,
        )

        # highlight dot same size
        ax.scatter(
            means[max_idx],
            max_idx,
            color="#2B5169FF",  # your highlight color
            s=90,
            zorder=4,
        )

        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=13)  # bigger ytick font
        ax.invert_yaxis()

        ax.set_xlim(0, max(means) * 1.15)
        if i == 2:
            ax.set_xlabel("Mean Dice", fontsize=14, fontweight="normal")
        ax.set_ylabel("")
        ax.annotate(
            param,
            xy=(0, 0.5),
            xycoords="axes fraction",
            fontsize=14,
            fontweight="normal",
            ha="right",
            va="center",
            rotation=90,  # keep vertical if desired
            xytext=(-35, 0),
            textcoords="offset points",
        )

        ax.tick_params(axis="x", labelsize=13)

        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=1.0)
        ax.xaxis.grid(False)

        sns.despine(ax=ax, top=True, right=True)

        for y_pos, mean_val in enumerate(means):
            ax.text(mean_val + 0.02, y_pos, f"{mean_val:.3f}", va="center", fontsize=11)

    plt.tight_layout()
    plt.show()


def unet_paper_aug_comparison(unet_dice, unet_augmented_dice):
    """
    Create a comparison plot for UNet and Augmented UNet Dice coefficients over epochs.
    Parameters:
    - unet_dice: List of Dice coefficients for UNet over epochs.
    - unet_augmented_dice: List of Dice coefficients for Augmented UNet over epochs.
    """
    # Set style and palette
    palette = sns.color_palette("crest", 4)
    font_title, font_label, font_ticks, font_legend = 19, 17, 15, 15

    # Create figure with 2 rows x 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # === Dice – Early (First 10 Epochs) ===
    epochs_early = range(1, 11)
    axes[0].plot(
        epochs_early,
        unet_dice[:10],
        marker="o",
        label="UNet",
        color=palette[0],
        linewidth=2,
    )
    axes[0].plot(
        epochs_early,
        unet_augmented_dice[:10],
        marker="o",
        label="Augmented UNet",
        color=palette[-1],
        linewidth=2,
    )
    axes[0].set_title("Dice – First 10 Epochs", fontsize=font_title)
    axes[0].set_ylabel("Dice Coefficient", fontsize=font_label)
    axes[0].tick_params(labelsize=font_ticks)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=font_legend)
    axes[0].set_xlabel("Epochs", fontsize=font_label)
    sns.despine(ax=axes[0])

    # === Dice – Full 40 Epochs ===
    epochs_full = range(1, 41)
    axes[1].plot(
        epochs_early, unet_dice, marker="o", label="UNet", color=palette[0], linewidth=2
    )
    axes[1].plot(
        epochs_full,
        unet_augmented_dice,
        marker="o",
        label="Augmented UNet",
        color=palette[-1],
        linewidth=2,
    )
    axes[1].set_title("Dice – Full 40 Epochs", fontsize=font_title)
    axes[1].tick_params(labelsize=font_ticks)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=font_legend)
    axes[1].set_xlabel("Epochs", fontsize=font_label)
    sns.despine(ax=axes[1])

    plt.show()


def unet_paper_results(unet_history, unet_aug_history):
    """
    Create a comparison plot for UNet and Augmented UNet Dice coefficients and F1 scores over epochs.
    Parameters:
    - unet_history: Dictionary containing 'loss', 'val_loss', 'dice_coef', and 'val_dice_coef' for UNet.
    - unet_aug_history: Dictionary containing 'loss', 'val_loss', 'dice_coef', and 'val_dice_coef' for Augmented UNet.
    """
    epochs = np.arange(1, 31)

    palette = sns.color_palette("crest", 4)

    # Figure and layout
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 5], wspace=0.15)

    # Left: Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        epochs,
        unet_history["loss"],
        label="Training Loss",
        color=palette[0],
        linewidth=2,
    )
    ax1.plot(
        epochs,
        unet_history["val_loss"],
        label="Validation Loss",
        color=palette[0],
        linestyle="--",
        linewidth=2,
    )
    ax1.plot(
        epochs,
        unet_aug_history["loss"],
        label="Augmented Training Loss",
        color=palette[-1],
        linewidth=2,
    )
    ax1.plot(
        epochs,
        unet_aug_history["val_loss"],
        label="Augmented Validation Loss",
        color=palette[-1],
        linestyle="--",
        linewidth=2,
    )
    ax1.set_title("Loss Comparison", fontsize=16, fontweight="normal")
    ax1.set_xlabel("Epochs", fontsize=14, fontweight="normal")
    ax1.set_ylabel("Loss", fontsize=14, fontweight="normal")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(frameon=False, fontsize=12)

    # Right: Dice
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        epochs,
        unet_history["dice_coef"],
        label="Training Dice",
        color=palette[0],
        linewidth=2,
    )
    ax2.plot(
        epochs,
        unet_history["val_dice_coef"],
        label="Validation Dice",
        color=palette[0],
        linestyle="--",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        unet_aug_history["dice_coef"],
        label="Augmented Training Dice",
        color=palette[-1],
        linewidth=2,
    )
    ax2.plot(
        epochs,
        unet_aug_history["val_dice_coef"],
        label="Augmented Validation Dice",
        color=palette[-1],
        linestyle="--",
        linewidth=2,
    )
    ax2.set_title("Dice Score Comparison", fontsize=16, fontweight="normal")
    ax2.set_xlabel("Epochs", fontsize=14, fontweight="normal")
    ax2.set_ylabel("Dice Coefficient", fontsize=14, fontweight="normal")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(frameon=False, fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()


def federated_initial_comparison(
    unet_history_0, unet_history_1, all_validation_loss, all_validation_dice
):
    """
    Create a comparison plot for Federated Learning results of two clients against their centralised UNet
    training results.
    Parameters:
    - unet_history_0: Dictionary containing 'val_loss' and 'val_dice_coef' for centralised training for a Client 1.
    - unet_history_1: Dictionary containing 'val_loss' and 'val_dice_coef' for centralised training for a Client 2.
    - all_validation_loss: List of validation loss for both clients in Federated Learning.
    - all_validation_dice: List of validation dice coefficients for both clients in Federated Learning.
    """
    epochs = np.arange(1, 31)

    palette = sns.color_palette("crest", 4)

    # Figure and layout
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 5], wspace=0.05)

    # Left: Loss
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(
        epochs,
        all_validation_loss[0],
        marker="o",
        label="Client 1 FedAvg Loss",
        color=palette[0],
        linewidth=2,
    )
    ax1.plot(
        epochs,
        all_validation_loss[1],
        marker="o",
        label="Client 2 FedAvg Loss",
        color=palette[-1],
        linewidth=2,
    )
    ax1.set_title(
        "Federated Loss - Client 1 vs Client 2", fontsize=19, fontweight="normal"
    )
    ax1.set_xlabel("Epochs", fontsize=17, fontweight="normal")
    ax1.set_ylabel("Loss", fontsize=17, fontweight="normal")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(frameon=False, fontsize=15)
    ax1.tick_params(labelsize=14)
    ax1.set_ylim(
        -0.1,
        max(
            max(all_validation_loss[0]),
            max(all_validation_loss[1]),
            max(unet_history_0["val_loss"]),
            max(unet_history_1["val_loss"]),
        )
        * 1.1,
    )

    # Right: Dice
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(
        epochs,
        all_validation_dice[0],
        marker="o",
        label="Client 1 FedAvg Dice",
        color=palette[0],
        linewidth=2,
    )
    ax2.plot(
        epochs,
        all_validation_dice[1],
        marker="o",
        label="Client 2 FedAvg Dice",
        color=palette[-1],
        linewidth=2,
    )
    ax2.set_title(
        "Federated Dice - Client 1 vs Client 2", fontsize=19, fontweight="normal"
    )
    ax2.set_xlabel("Epochs", fontsize=17, fontweight="normal")
    ax2.set_ylabel("Dice Coefficient", fontsize=17, fontweight="normal")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(frameon=False, fontsize=15)
    ax2.tick_params(labelsize=14)
    ax2.set_ylim(
        -0.1,
        max(
            max(all_validation_dice[0]),
            max(all_validation_dice[1]),
            max(unet_history_0["val_dice_coef"]),
            max(unet_history_1["val_dice_coef"]),
        )
        * 1.1,
    )

    ax3 = fig.add_subplot(gs[0, 0])
    ax3.plot(
        epochs,
        unet_history_0["val_loss"],
        marker="o",
        label="Client 1 Centralised Loss",
        color=palette[0],
        linewidth=2,
    )
    ax3.plot(
        epochs,
        unet_history_1["val_loss"],
        marker="o",
        label="Client 2 Centralised Loss",
        color=palette[-1],
        linewidth=2,
    )
    ax3.set_title(
        "Centralised Loss - Client 1 vs Client 2", fontsize=19, fontweight="normal"
    )
    ax3.set_xlabel("Epochs", fontsize=17, fontweight="normal")
    ax3.set_ylabel("Loss", fontsize=17, fontweight="normal")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend(frameon=False, fontsize=15)
    ax3.tick_params(labelsize=14)
    ax3.set_ylim(
        -0.1,
        max(
            max(all_validation_loss[0]),
            max(all_validation_loss[1]),
            max(unet_history_0["val_loss"]),
            max(unet_history_1["val_loss"]),
        )
        * 1.1,
    )

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(
        epochs,
        unet_history_0["val_dice_coef"],
        marker="o",
        label="Client 1 Centralised Dice",
        color=palette[0],
        linewidth=2,
    )
    ax4.plot(
        epochs,
        unet_history_1["val_dice_coef"],
        marker="o",
        label="Client 2 Centralised Dice",
        color=palette[-1],
        linewidth=2,
    )
    ax4.set_title(
        "Centralised Dice - Client 1 vs Client 2", fontsize=19, fontweight="normal"
    )
    ax4.set_xlabel("Epochs", fontsize=17, fontweight="normal")
    ax4.set_ylabel("Dice Coefficient", fontsize=17, fontweight="normal")
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend(frameon=False, fontsize=15)
    ax4.tick_params(labelsize=14)
    ax4.set_ylim(
        -0.1,
        max(
            max(all_validation_dice[0]),
            max(all_validation_dice[1]),
            max(unet_history_0["val_dice_coef"]),
            max(unet_history_1["val_dice_coef"]),
        )
        * 1.1,
    )

    fig.align_ylabels([ax1, ax2])  # Align loss axes
    fig.align_ylabels([ax3, ax4])  # Align dice axes

    sns.despine()
    plt.show()


def fed_2_comparison(
    all_validation_loss,
    all_validation_dice,
    all_validation_loss_prox,
    all_validation_dice_prox,
    client1_final,
    client2_final,
    epochs,
    epochs1,
):
    """
    Create a comparison plot for Federated Learning results of two clients - FedAvg vs FedProx.
    Parameters:
    - all_validation_loss: List of validation loss for both clients in Federated Learning using FedAvg.
    - all_validation_dice: List of validation dice coefficients for both clients in Federated Learning using FedAvg.
    - all_validation_loss_prox: List of validation loss for both clients in Federated Learning using FedProx.
    - all_validation_dice_prox: List of validation dice coefficients for both clients in Federated Learning using FedProx.
    """
    palette = sns.color_palette("crest", 6)

    fs, ts, ls = 18, 20, 14
    # Setup 2 rows × 3 columns
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[5, 5, 3], wspace=0.05)

    # ===== Client 1 Loss =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        epochs,
        all_validation_loss[0],
        marker="o",
        label="Client 1 FedAvg Loss",
        color=palette[0],
        linewidth=2,
    )
    ax1.plot(
        epochs,
        all_validation_loss_prox[0],
        marker="x",
        label="Client 1 FedProx Loss",
        color=palette[-1],
        linewidth=2,
    )
    ax1.set_title("Client 1 - Validation Loss", fontsize=ts)
    ax1.set_xlabel("Epochs", fontsize=fs)
    ax1.set_ylabel("Loss", fontsize=fs)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(frameon=False, fontsize=15)
    ax1.tick_params(labelsize=ls)
    ax1.set_ylim(
        -0.1, max(max(all_validation_loss[0]), max(all_validation_loss_prox[0])) * 1.1
    )

    # ===== Client 1 Dice =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        epochs,
        all_validation_dice[0],
        marker="o",
        label="Client 1 FedAvg Dice",
        color=palette[0],
        linewidth=2,
    )
    ax2.plot(
        epochs,
        all_validation_dice_prox[0],
        marker="x",
        label="Client 1 FedProx Dice",
        color=palette[-1],
        linewidth=2,
    )
    ax2.set_title("Client 1 - Validation Dice", fontsize=ts)
    ax2.set_xlabel("Epochs", fontsize=fs)
    ax2.set_ylabel("Dice Coefficient", fontsize=fs)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(frameon=False, fontsize=15)
    ax2.tick_params(labelsize=ls)
    ax2.set_ylim(
        -0.1, max(max(all_validation_dice[0]), max(all_validation_dice_prox[0])) * 1.1
    )

    # ===== Client 1 - Final Dice Bar Plot =====
    ax3 = fig.add_subplot(gs[0, 2])
    labels = ["FedAvg", "FedProx", "Cent."]
    x = np.arange(len(labels))
    width = 0.35

    ax3.bar(
        x,
        client1_final,
        width=width,
        label="Final",
        color=[palette[0], palette[-1], palette[3]],
        alpha=0.7,
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Dice Coefficient", fontsize=fs)
    ax3.set_title("Client 1 - Final Test Dice", fontsize=ts)
    ax3.tick_params(axis="y", labelsize=ls)
    ax3.tick_params(axis="x", labelsize=fs)
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    sns.despine(ax=ax3)
    for label in ax3.get_xticklabels():
        label.set_y(-0.05)

    for i, v in enumerate(client1_final):
        ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=ls)

    # ===== Client 2 Loss =====
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(
        epochs1,
        all_validation_loss[1],
        marker="o",
        label="Client 2 FedAvg Loss",
        color=palette[0],
        linewidth=2,
    )
    ax4.plot(
        epochs1,
        all_validation_loss_prox[1],
        marker="x",
        label="Client 2 FedProx Loss",
        color=palette[-1],
        linewidth=2,
    )
    ax4.set_title("Client 2 - Validation Loss", fontsize=ts)
    ax4.set_xlabel("Epochs", fontsize=fs)
    ax4.set_ylabel("Loss", fontsize=fs)
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.tick_params(labelsize=ls)
    ax4.legend(frameon=False, fontsize=15)
    ax4.set_ylim(
        -0.1, max(max(all_validation_loss[1]), max(all_validation_loss_prox[1])) * 1.1
    )

    # ===== Client 2 Dice =====
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(
        epochs1,
        all_validation_dice[1],
        marker="o",
        label="Client 2 FedAvg Dice",
        color=palette[0],
        linewidth=2,
    )
    ax5.plot(
        epochs1,
        all_validation_dice_prox[1],
        marker="x",
        label="Client 2 FedProx Dice",
        color=palette[-1],
        linewidth=2,
    )
    ax5.set_title("Client 2 - Validation Dice", fontsize=ts)
    ax5.set_xlabel("Epochs", fontsize=fs)
    ax5.set_ylabel("Dice Coefficient", fontsize=fs)
    ax5.grid(True, linestyle="--", alpha=0.7)
    ax5.tick_params(labelsize=ls)
    ax5.legend(frameon=False, fontsize=15)
    ax5.set_ylim(
        -0.1, max(max(all_validation_dice[1]), max(all_validation_dice_prox[1])) * 1.1
    )

    # ===== Client 2 - Final Dice Bar Plot =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(
        x,
        client2_final,
        width=width,
        label="Final",
        color=[palette[0], palette[-1], palette[3]],
        alpha=0.7,
    )
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.set_ylim(0, 1)
    ax6.set_ylabel("Dice Coefficient", fontsize=fs)
    ax6.set_title("Client 2 - Final Test Dice", fontsize=ts)
    ax6.tick_params(axis="y", labelsize=ls)
    ax6.tick_params(axis="x", labelsize=fs)
    ax6.grid(axis="y", linestyle="--", alpha=0.7)
    sns.despine(ax=ax6)
    for label in ax6.get_xticklabels():
        label.set_y(-0.08)

    for i, v in enumerate(client2_final):
        ax6.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=ls)

    sns.despine()
    plt.show()


def fed_2_global(
    all_losses, all_dices, all_losses_prox, all_dices_prox, best_dices, final_dices
):
    """
    Create a comparison plot for global models in Federated Learning - FedAvg vs FedProx.
    Parameters:
    - all_losses: List of validation loss for global models in Federated Learning using FedAvg.
    - all_dices: List of validation dice coefficients for global models in Federated Learning using FedAvg.
    - all_losses_prox: List of validation loss for global models in Federated Learning using FedProx.
    - all_dices_prox: List of validation dice coefficients for global models in Federated Learning using FedProx.
    """
    # Example epoch range (e.g. every 3 epochs up to 30)
    epochs = np.arange(1, 11)

    # Style
    palette = sns.color_palette("crest", 4)
    font_title, font_label, font_ticks, font_legend = 19, 17, 14, 15

    # -- best vs final scores --
    methods = ["FedAvg", "FedProx", "Centralised"]
    x = np.arange(len(methods))
    bar_width = 0.4

    # Create figure with 3 subplots (2 line plots + 1 grouped bar plot)
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 1]}, sharex=False
    )

    # === Loss over Epochs ===
    axes[0].plot(
        epochs, all_losses, label="FedAvg", color=palette[0], linewidth=2, marker="o"
    )
    axes[0].plot(
        epochs,
        all_losses_prox,
        label="FedProx",
        color=palette[-1],
        linewidth=2,
        marker="s",
    )
    axes[0].set_title("Global Models - Test Loss", fontsize=font_title, pad=15)
    axes[0].set_xlabel("Rounds", fontsize=font_label)
    axes[0].set_ylabel("Loss", fontsize=font_label)
    axes[0].tick_params(labelsize=font_ticks)
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend(frameon=False, fontsize=font_legend)
    sns.despine(ax=axes[0])

    # === Dice over Epochs ===
    axes[1].plot(
        epochs, all_dices, label="FedAvg", color=palette[0], linewidth=2, marker="o"
    )
    axes[1].plot(
        epochs,
        all_dices_prox,
        label="FedProx",
        color=palette[-1],
        linewidth=2,
        marker="s",
    )
    axes[1].set_title("Global Models - Test Dice", fontsize=font_title, pad=15)
    axes[1].set_xlabel("Rounds", fontsize=font_label)
    axes[1].set_ylabel("Dice Coefficient", fontsize=font_label)
    axes[1].tick_params(labelsize=font_ticks)
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend(frameon=False, fontsize=font_legend)
    sns.despine(ax=axes[1])

    # === Grouped Barplot – Best vs Final Dice ===
    palette1 = sns.color_palette("crest", 6)
    bar_colors = [palette[0], palette[-1], palette1[3]]  # FedAvg, FedProx, Cent.
    ax = axes[2]

    #  '//' for first bar, '//' for second bar
    bars_best = ax.bar(
        x - bar_width / 2,
        best_dices,
        width=bar_width,
        label="Best Model",
        color=bar_colors,
        linewidth=0.5,
        hatch="//",
    )
    bars_final = ax.bar(
        x + bar_width / 2,
        final_dices,
        width=bar_width,
        label="Final Model",
        color=bar_colors,
        linewidth=0.5,
        alpha=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_title("Best vs Final Models – Test Dice", fontsize=font_title, pad=15)
    ax.set_ylabel("Dice Coefficient", fontsize=font_label)
    # tick only the bottom
    ax.tick_params(axis="x", labelsize=font_label)
    ax.tick_params(axis="y", labelsize=font_ticks)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(frameon=False, fontsize=font_legend)
    sns.despine(ax=ax)

    # Align x-axis tick labels
    for label in ax.get_xticklabels():
        label.set_verticalalignment("top")
        label.set_y(-0.07)

    # Add value annotations
    for bars in [bars_best, bars_final]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=font_legend,
            )

    import matplotlib.patches as mpatches

    type_patches = [
        mpatches.Patch(
            facecolor="grey", alpha=1.0, label="Best", edgecolor="k", hatch="//"
        ),
        mpatches.Patch(facecolor="grey", alpha=0.4, label="Final", edgecolor="k"),
    ]
    ax.legend(
        handles=type_patches, loc="upper right", fontsize=font_legend, frameon=False
    )

    plt.tight_layout()
    plt.show()

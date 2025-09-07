# Configuration for the pretrained model training script

base_path = "/home/mn628/FEDERATED_LEARNING/data/"  # Data path

MODEL_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/pretrained_model/segmentation_model.h5"  # Final model
BEST_MODEL = "/home/mn628/FEDERATED_LEARNING/mn628/results/pretrained_model/best_fpn_model.h5"  # Best model

RESULTS_IMAGE_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/pretrained_model/fpn_segmentation_results.png"
HISTORY_TXT_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/pretrained_model/fpn_segmentation_history.txt"
CONF_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/pretrained_model/fpn_segmentation_config.txt"

BS = 16  # Batch size
LR = 1e-4  # Original learning rate
LR_FINETUNING = 1e-6  # Smaller learning rate for fine-tuning
patience = 7  # Early stopping patience
NUM_EPOCHS = 30  # Number of epochs for initial training
NUM_EPOCHS_FINETUNING = 15  # Number of epochs for fine-tuning

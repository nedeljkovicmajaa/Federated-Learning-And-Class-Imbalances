"Path to the dataset"

DATA_PATH = "/home/mn628/FEDERATED_LEARNING/data/"

"UNet parameters - model + training configuration"
num_filters = 8
dropout = 0.3
batch_norm = True
patience = 5
early_stopping = False  # Use early stopping during training

BS = 128  # Batch size
EPOCHS = 30  # Number of epochs
AUG = False  # Data augmentation
LR = 0.001  # Learning rate for the optimizer


"""Train on a subset of the data"""
SUBSET = True
NUM_CLIENTS = 2
percent = 0.7
PROBLEM_TYPE = "statstical_het/"  # 'statstical_het/', 'system_het/' or just 'normal/'
CLIENT_ID = 0


"Path to save model and training history"
MODEL_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_segmentation/sss/unet_model.h5"
HISTORY_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_segmentation/sss/unet_history.png"
HISTORY_TXT_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_segmentation/sss/unet_history.txt"
SAMPLES_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_segmentation/sss/unet_samples.png"
CONF_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_segmentation/sss/unet_config.txt"

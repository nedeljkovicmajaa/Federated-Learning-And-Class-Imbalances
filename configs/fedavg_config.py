# Paths to the dataset and folder to the federated learning implementation
ALL_DATA_PATH = "/home/mn628/FEDERATED_LEARNING/"
INITIAL_FED_PATH = "/home/mn628/FEDERATED_LEARNING/mn628/results/seg_fedavg/"


PROBLEM_TYPE = "normal/"  # 'statstical_het/', 'system_het/' or just 'normal/'
chosen_dataset = (
    "data"  # 'data' for the full dataset or 'data_subset' for a subset
)

NUM_EPOCHS = (
    3  # Number of epochs for training - for default or statistical heterogeneity
)
BS = 128  # Batch size for training
AUG = True  # Whether to apply data augmentation
percent = (
    0.5  # Percentage of data for client 0 in the split - for statistical heterogeneity
)
BASE_NUM_FILT = 8  # Number of filters in the first layer
DROPOUT = 0.3  # Dropout rate
NUM_ROUNDS = 10  # Number of rounds for federated learning
NUM_CLIENTS = 2  # Number of clients in the federated learning setup

SAMPLES_PATH = (
    INITIAL_FED_PATH + PROBLEM_TYPE + "sample.png"
)  # Path to the sample image for visualization
DATA_PATH = ALL_DATA_PATH + chosen_dataset  # Path to the dataset

hostname = "10.43.77.230"  # Hostname of the server

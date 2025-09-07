"Fixed parameters"

MEAN, STD = 0.1307, 0.3081  # Mean and std for MNIST dataset (known values)
DATASET_PATH = "/home/mn628/FEDERATED_LEARNING/data"  # Dataset path

"Parameters for centralised learning"
train_val_split = 0.8  # proportion for train/val split

"Parameters for both centralised and federated learning"
BS = 64  # Batch size for training and validation
LR = 0.01  # Learning rate
momentum = 0.9  # Momentum for SGD optimizer

"Parameters for federated learning"
NUM_ROUNDS = 3  # Number of rounds for federated learning

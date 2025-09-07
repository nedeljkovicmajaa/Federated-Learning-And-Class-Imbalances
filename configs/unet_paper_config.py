import torch

# Data preparation
im_size = 128  # image size
problem_pos = 4454  # corrupted image position

train_path = (
    "/home/mn628/FEDERATED_LEARNING/new_database/seg/train"  # path to the training data
)
test_path = "/home/mn628/FEDERATED_LEARNING/new_database/seg/test"
val_path = "/home/mn628/FEDERATED_LEARNING/new_database/seg/val"

train_save = "/home/mn628/FEDERATED_LEARNING/data/train_images.npy"  # path to save the training data
train_mask_save = "/home/mn628/FEDERATED_LEARNING/data/train_masks.npy"
val_save = "/home/mn628/FEDERATED_LEARNING/data/val_images.npy"
val_mask_save = "/home/mn628/FEDERATED_LEARNING/data/val_masks.npy"
test_save = "/home/mn628/FEDERATED_LEARNING/data/test_images.npy"
test_mask_save = "/home/mn628/FEDERATED_LEARNING/data/test_masks.npy"

# Model and training parameters
TESTING = False  # testing mode - only 10 samples are used

save_weights_path = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_paper/save_weights/best_model_TEST.pth"
results_file = (
    "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_paper/results/results_TEST.txt"
)

base_num_filters = 32  # number of filters in the first layer
batch_size = 64
init_num_classes = 1  # number of classes without background
lr = 0.01
num_epochs = 5
momentum = 0.9
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_freq = 10  # frequency of printing the results

# Prediction parameters
weights_path = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_paper/save_weights/best_model_augmented_TEST.pth"
prediction_path = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_paper/examples/prediction_aug_TEST.png"
label_path = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_paper/examples/label_aug_TEST.png"

# Evaluation path
multiple_predictions_path = "/home/mn628/FEDERATED_LEARNING/mn628/results/UNet_paper/examples/predictions_TEST.png"

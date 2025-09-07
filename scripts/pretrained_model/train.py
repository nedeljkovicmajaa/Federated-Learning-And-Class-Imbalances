import time

from configs.pretrained_config import *
from src.pretrained_model.train import *


def main():
    x_train, y_train, x_val, y_val = load_data()

    # Start timing the training process
    time_start = time.time()

    model = build_model()
    trained_model = train_model(model, x_train, y_train, x_val, y_val)

    # End timing the training process
    time_end = time.time()
    # Calculate total duration and save logs
    total_duration = time_end - time_start
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_duration))
    log_lines = [
        "\nUNet Segmentation Training Completed",
        f"Total time: {total_duration:.2f} seconds",
        f"Total time (formatted): {formatted_time}",
    ]
    print("\n".join(log_lines))
    with open(CONF_PATH, "a") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Load test data and evaluate the model on test set
    test_images = np.load(base_path + "test_images.npy")
    test_masks = np.load(base_path + "test_masks.npy")

    trained_model.save(MODEL_PATH)
    evaluate_model(trained_model, test_images, test_masks)


if __name__ == "__main__":
    main()

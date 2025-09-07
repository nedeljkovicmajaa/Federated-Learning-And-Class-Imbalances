from configs.pretrained_config import *
from src.pretrained_model.evaluate import *


def main():
    # Load model
    model = load_trained_model(BEST_MODEL)

    # Load data
    train_images, train_masks = load_and_preprocess_data()

    # Evaluate
    print("\nMetrics:")
    evaluate_model(model, train_images, train_masks)


if __name__ == "__main__":
    main()

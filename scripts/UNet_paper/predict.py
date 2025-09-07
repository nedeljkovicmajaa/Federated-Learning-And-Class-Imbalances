import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from configs.unet_paper_config import *
from src.UNet_paper.unet import UNet


def one_prediction(model, test_images, test_masks, save=True):
    """
    Function to perform a single prediction on a random test image and save the output mask and ROI label.
    """
    # select a random image for testing
    check = np.random.randint(0, test_images.shape[0], 1)[0]
    test_images = np.transpose(test_images, (0, 3, 1, 2))[check]
    test_masks = np.transpose(test_masks, (0, 3, 1, 2))[check]

    # convert the image and mask to tensors
    img = torch.tensor(test_images).unsqueeze(0).float()
    roi_img = torch.tensor(test_masks).float()

    with torch.no_grad():

        init_img = torch.zeros(img.shape, device=device)
        model(init_img)  # model initialisation

        output = model(img.to(device))  # run the image through the model

        # post-processing: Convert the output to predicted class labels (argmax)
        prediction = output["out"].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction[prediction == 1] = 255  # for visualization

        # process the ROI mask similarly
        roi_img = roi_img.squeeze(0)
        roi_img = roi_img.to("cpu").numpy().astype(np.uint8)
        roi_img[roi_img == 1] = 255
        roi_img = Image.fromarray(roi_img)

        # save the prediction and true label
        mask = Image.fromarray(prediction)

        if save:
            mask.save(prediction_path)
            roi_img.save(label_path)

    return img, prediction, roi_img


def main():
    # create and load the model
    model = UNet(in_channels=1, num_classes=init_num_classes + 1, base_c=32)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # load the test images and masks
    test_images, test_masks = np.load(test_save), np.load(test_mask_save)

    model.eval()

    img_1, prediction_1, roi_img_1 = one_prediction(
        model, test_images, test_masks, save=True
    )
    img2, prediction_2, roi_img_2 = one_prediction(
        model, test_images, test_masks, save=True
    )
    img3, prediction_3, roi_img_3 = one_prediction(
        model, test_images, test_masks, save=True
    )

    plt.figure(figsize=(20, 20))
    plt.subplot(3, 3, 1), plt.imshow(img_1[0, 0].cpu().numpy(), cmap="gray"), plt.title(
        "Test Image 1", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 2), plt.imshow(prediction_1, cmap="gray"), plt.title(
        "Prediction 1", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 3), plt.imshow(roi_img_1, cmap="gray"), plt.title(
        "Ground Truth 1", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 4), plt.imshow(img2[0, 0].cpu().numpy(), cmap="gray"), plt.title(
        "Test Image 2", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 5), plt.imshow(prediction_2, cmap="gray"), plt.title(
        "Prediction 2", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 6), plt.imshow(roi_img_2, cmap="gray"), plt.title(
        "Ground Truth 2", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 7), plt.imshow(img3[0, 0].cpu().numpy(), cmap="gray"), plt.title(
        "Test Image 3", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 8), plt.imshow(prediction_3, cmap="gray"), plt.title(
        "Prediction 3", fontsize=25
    ), plt.axis("off")
    plt.subplot(3, 3, 9), plt.imshow(roi_img_3, cmap="gray"), plt.title(
        "Ground Truth 3", fontsize=25
    ), plt.axis("off")
    plt.tight_layout()
    plt.savefig(multiple_predictions_path)


if __name__ == "__main__":
    main()

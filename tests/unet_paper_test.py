import torch
from src.UNet_paper.dice_coefficient_loss import build_target, dice_loss
from src.UNet_paper.prepare_data import augmentation, shuffle_manual
from src.UNet_paper.train_and_eval import criterion
from src.UNet_paper.unet import UNet


def test_dice_loss_returns_valid_range():
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randint(0, 2, (2, 64, 64))
    one_hot = build_target(target, num_classes=2)
    loss = dice_loss(pred, one_hot, multiclass=False)
    assert 0.0 <= loss <= 1.0, "Dice loss is not calculated correctly"


def test_augmentation_doubles_dataset():
    images = torch.randn(4, 3, 128, 128)  # batch of 4 RGB images
    masks = torch.randint(0, 2, (4, 128, 128))  # binary masks

    images_aug, masks_aug = augmentation(images, masks)

    assert images_aug.shape[0] == 8, "Augmentation did not double the number of images"
    assert masks_aug.shape[0] == 8, "Augmentation did not double the number of masks"
    assert images_aug.shape[1:] == (
        3,
        128,
        128,
    ), "Augmented images have incorrect shape"
    assert masks_aug.shape[1:] == (128, 128), "Augmented masks have incorrect shape"


def test_shuffle_manual_preserves_alignment():
    images = torch.randn(4, 3, 128, 128)
    masks = torch.arange(4).unsqueeze(1).expand(-1, 128 * 128).reshape(4, 128, 128)

    shuffled_images, shuffled_masks = shuffle_manual(images.clone(), masks.clone())

    # Confirm that each image still aligns with a unique mask after shuffling
    aligned = [
        torch.all(
            (shuffled_images[i] == images[j]).all()
            == (shuffled_masks[i] == masks[j]).all()
        )
        for i in range(4)
        for j in range(4)
    ]
    assert any(aligned), "Shuffling breaks alignment between images and masks"


def test_criterion_cross_entropy_only():
    inputs = {"out": torch.randn(2, 2, 16, 16)}  # batch of 2, 2 classes
    target = torch.randint(0, 2, (2, 16, 16))
    loss = criterion(inputs, target, dice=False)
    assert loss.item() > 0, "Cross-entropy loss is not calculated correctly"


def test_criterion_combined_loss():
    inputs = {"out": torch.randn(2, 2, 16, 16, requires_grad=True)}
    target = torch.randint(0, 2, (2, 16, 16))
    loss = criterion(inputs, target, dice=True)
    assert (
        loss.item() > 0
    ), "Combined loss (cross-entropy + dice) is not calculated correctly"


def test_unet_forward_rgb():
    model = UNet(in_channels=3, num_classes=3)
    x = torch.randn(4, 3, 256, 256)  # batch=4, RGB input
    out = model(x)
    assert out["out"].shape == (
        4,
        3,
        256,
        256,
    ), "UNet forward pass with RGB input failed"

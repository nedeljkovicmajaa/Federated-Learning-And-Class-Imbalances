import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1):
    """
    Compute the Dice coefficient for binary segmentation.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_coef_loss(y_true, y_pred):
    """
    Compute the Dice loss.
    """
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice_loss = dice_coef_loss(y_true, y_pred)
    return bce + dice_loss


def focal_loss(y_true, y_pred, delta=0.6, gamma=0.5):
    """
    Unified Focal Loss for medical image segmentation.
    Combines asymmetric focal loss and Dice loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)  # Clip for stability

    # Asymmetric Focal Loss
    cross_entropy = -(
        y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    )
    focal_term = tf.where(y_true == 1, tf.pow(1 - y_pred, gamma), tf.pow(y_pred, gamma))
    asymmetric_focal = focal_term * cross_entropy

    # Dice Loss
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

    # Unified Loss (δ controls focal vs. Dice weighting)
    return (1 - delta) * tf.reduce_mean(asymmetric_focal) + delta * dice_loss


def iou(y_true, y_pred, smooth=1):
    """
    Compute the Intersection over Union (IoU) for binary segmentation.
    """
    iou = dice_coef(y_true, y_pred, smooth)
    iou = iou / (2 - iou)
    return iou

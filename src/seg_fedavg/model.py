import tensorflow as tf

from configs.fedavg_config import *


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def iou(y_true, y_pred, smooth=1):
    dice = dice_coef(y_true, y_pred, smooth)
    return dice / (2 - dice)


class UNetModel:
    def __init__(
        self,
        input_shape=(128, 128, 1),
        num_filters=BASE_NUM_FILT,
        dropout_rate=DROPOUT,
        do_batch_norm=True,
    ):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.do_batch_norm = do_batch_norm
        self.model = self.build_model()

    def conv2d_block(self, input_tensor, num_filters, kernel_size=3):
        x = tf.keras.layers.Conv2D(
            num_filters, kernel_size, padding="same", kernel_initializer="he_normal"
        )(input_tensor)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            num_filters, kernel_size, padding="same", kernel_initializer="he_normal"
        )(x)
        if self.do_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def encoder_block(self, input_tensor, num_filters):
        x = self.conv2d_block(input_tensor, num_filters)
        p = tf.keras.layers.MaxPooling2D((2, 2))(x)
        p = tf.keras.layers.Dropout(self.dropout_rate)(p)
        return x, p

    def decoder_block(self, input_tensor, skip_tensor, num_filters):
        x = tf.keras.layers.Conv2DTranspose(
            num_filters, (3, 3), strides=2, padding="same"
        )(input_tensor)
        x = tf.keras.layers.concatenate([x, skip_tensor])
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = self.conv2d_block(x, num_filters)
        return x

    def build_model(self):
        inputs = tf.keras.layers.Input(self.input_shape)

        c1, p1 = self.encoder_block(inputs, self.num_filters)
        c2, p2 = self.encoder_block(p1, self.num_filters * 2)
        c3, p3 = self.encoder_block(p2, self.num_filters * 4)
        c4, p4 = self.encoder_block(p3, self.num_filters * 8)

        c5 = self.conv2d_block(p4, self.num_filters * 16)

        c6 = self.decoder_block(c5, c4, self.num_filters * 8)
        c7 = self.decoder_block(c6, c3, self.num_filters * 4)
        c8 = self.decoder_block(c7, c2, self.num_filters * 2)
        c9 = self.decoder_block(c8, c1, self.num_filters)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)
        model = tf.keras.Model(inputs, outputs)
        return model

    def compile(self):
        self.model.compile(
            optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef, iou]
        )

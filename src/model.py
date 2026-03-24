import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose,
    concatenate, BatchNormalization, Activation,
)


def conv_block(inputs, n_filters=32, dropout_prob=0.0, max_pooling=True):
    """
    Contracting block: Conv → BN → ReLU → Conv → BN → ReLU [→ Dropout] [→ MaxPool].

    Returns (next_layer, skip_connection).
    """
    x = Conv2D(
        n_filters, kernel_size=3, padding='same',
        kernel_initializer='he_normal',
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(
        n_filters, kernel_size=3, padding='same',
        kernel_initializer='he_normal',
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)

    skip_connection = x

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(x)
    else:
        next_layer = x

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32, dropout_prob=0.0):
    """
    Expanding block: ConvTranspose → Concat(skip) → Conv → BN → ReLU → Conv → BN → ReLU.
    """
    x = Conv2DTranspose(
        n_filters, kernel_size=3, strides=(2, 2), padding='same',
        kernel_initializer='he_normal',
    )(expansive_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, contractive_input], axis=3)

    x = Conv2D(
        n_filters, kernel_size=3, padding='same',
        kernel_initializer='he_normal',
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(
        n_filters, kernel_size=3, padding='same',
        kernel_initializer='he_normal',
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)

    return x


def build_unet_model(input_shape=(256, 256, 3), n_filters=32, n_classes=23):
    """
    Improved U-Net with BatchNormalization and broader dropout.

    Args:
        input_shape: (H, W, C) tuple.
        n_filters: Base filter count (doubled at each encoder level).
        n_classes: Number of segmentation classes.
    """
    inputs = Input(input_shape)

    # ── Encoder (contracting path) ──
    c1_pool, c1_skip = conv_block(inputs, n_filters, dropout_prob=0.1)
    c2_pool, c2_skip = conv_block(c1_pool, n_filters * 2, dropout_prob=0.1)
    c3_pool, c3_skip = conv_block(c2_pool, n_filters * 4, dropout_prob=0.2)
    c4_pool, c4_skip = conv_block(c3_pool, n_filters * 8, dropout_prob=0.3)

    # ── Bottleneck ──
    bn_out, _ = conv_block(c4_pool, n_filters * 16, dropout_prob=0.4, max_pooling=False)

    # ── Decoder (expanding path) ──
    u6 = upsampling_block(bn_out, c4_skip, n_filters * 8, dropout_prob=0.2)
    u7 = upsampling_block(u6, c3_skip, n_filters * 4, dropout_prob=0.15)
    u8 = upsampling_block(u7, c2_skip, n_filters * 2, dropout_prob=0.1)
    u9 = upsampling_block(u8, c1_skip, n_filters, dropout_prob=0.1)

    # ── Head ──
    x = Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(u9)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output logits (used with SparseCategoricalCrossentropy(from_logits=True))
    outputs = Conv2D(n_classes, kernel_size=1, padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet')
    return model

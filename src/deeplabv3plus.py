"""
DeepLabV3+ with a pretrained ResNet50 backbone for semantic segmentation.

Key advantages over vanilla U-Net:
  - **Transfer learning**: ImageNet-pretrained encoder captures rich features.
  - **Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale context without
    losing resolution, crucial for drone imagery with diverse object sizes.
  - **Decoder with low-level feature fusion**: Recovers sharp boundaries.

Reference: Chen et al., "Encoder-Decoder with Atrous Separable Convolution
           for Semantic Image Segmentation", ECCV 2018.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ── ASPP Module ──────────────────────────────────────────────────────────────

def _aspp(x, out_filters=256):
    """
    Atrous Spatial Pyramid Pooling.

    Applies parallel atrous convolutions at multiple rates plus image-level
    pooling, then concatenates them and projects to *out_filters*.
    """
    shape = tf.shape(x)
    h, w = shape[1], shape[2]

    # Branch 1: 1×1 convolution
    b1 = layers.Conv2D(out_filters, 1, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)

    # Branch 2-4: 3×3 atrous convolutions at rate 6, 12, 18
    branches = [b1]
    for rate in (6, 12, 18):
        b = layers.Conv2D(
            out_filters, 3, padding='same', dilation_rate=rate, use_bias=False,
        )(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('relu')(b)
        branches.append(b)

    # Branch 5: image-level (global average pooling → 1×1 conv → upsample)
    b5 = layers.GlobalAveragePooling2D(keepdims=True)(x)
    b5 = layers.Conv2D(out_filters, 1, padding='same', use_bias=False)(b5)
    b5 = layers.BatchNormalization()(b5)
    b5 = layers.Activation('relu')(b5)
    b5 = layers.Lambda(lambda t: tf.image.resize(t, (h, w)))(b5)
    branches.append(b5)

    # Concat + project
    concat = layers.Concatenate()(branches)
    out = layers.Conv2D(out_filters, 1, padding='same', use_bias=False)(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    out = layers.Dropout(0.5)(out)

    return out


# ── Decoder ──────────────────────────────────────────────────────────────────

def _decoder(aspp_out, low_level_feat, n_classes):
    """
    DeepLabV3+ decoder: upsample ASPP output, fuse with low-level features,
    then predict class logits.
    """
    # Project low-level features to 48 channels
    low = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
    low = layers.BatchNormalization()(low)
    low = layers.Activation('relu')(low)

    # Upsample ASPP to match low-level spatial dims
    target_shape = tf.shape(low)
    up = layers.Lambda(
        lambda t: tf.image.resize(t, (target_shape[1], target_shape[2]))
    )(aspp_out)

    # Concatenate
    x = layers.Concatenate()([up, low])

    # Refine
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)

    # Final logits
    x = layers.Conv2D(n_classes, 1, padding='same')(x)

    return x


# ── Public builder ───────────────────────────────────────────────────────────

def build_deeplabv3plus(input_shape=(256, 256, 3), n_classes=23, freeze_backbone=False):
    """
    Build a DeepLabV3+ model with a ResNet50 backbone pretrained on ImageNet.

    Args:
        input_shape: (H, W, 3).  H and W should be divisible by 16.
        n_classes: Number of segmentation classes.
        freeze_backbone: If True the ResNet50 weights are not updated during
            training (useful for very small datasets or initial warm-up).

    Returns:
        A compiled-ready ``tf.keras.Model`` outputting logits of shape
        (B, H, W, n_classes).
    """
    inputs = layers.Input(shape=input_shape)

    # ── Backbone ─────────────────────────────────────────────────────────
    backbone = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs,
    )
    if freeze_backbone:
        backbone.trainable = False

    # Low-level features from early stage (stride 4, e.g. conv2_block3_out)
    low_level_feat = backbone.get_layer('conv2_block3_out').output   # /4
    # High-level features from final stage (stride 16)
    high_level_feat = backbone.get_layer('conv4_block6_out').output  # /16

    # ── ASPP on high-level features ──────────────────────────────────────
    aspp_out = _aspp(high_level_feat, out_filters=256)

    # ── Decoder ──────────────────────────────────────────────────────────
    logits_small = _decoder(aspp_out, low_level_feat, n_classes)

    # ── Upsample logits to input resolution ──────────────────────────────
    logits = layers.Lambda(
        lambda t: tf.image.resize(t, (input_shape[0], input_shape[1]))
    )(logits_small)

    model = Model(inputs=inputs, outputs=logits, name='DeepLabV3Plus')
    return model

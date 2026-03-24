"""
Two-phase training script for DeepLabV3+ on the Semantic Drone Dataset.

Phase 1 — **Warm-up**: backbone frozen, only the ASPP + decoder are trained
          with a higher learning rate.  This prevents catastrophic forgetting
          of ImageNet features when the head is still random.

Phase 2 — **Fine-tune**: the entire model (backbone included) is unfrozen and
          trained end-to-end with a lower learning rate.
"""

import os
import tensorflow as tf
from .dataset import build_dataset
from .deeplabv3plus import build_deeplabv3plus


def train_deeplabv3plus(
    images_dir,
    masks_dir,
    epochs_warmup=20,
    epochs_finetune=80,
    batch_size=8,
    buffer_size=500,
    target_size=(256, 256),
    n_classes=23,
    validation_split=0.2,
    output_dir='data/output/deeplabv3plus',
):
    """
    Train DeepLabV3+ in two phases (warm-up → fine-tune).

    Returns:
        (model, history_warmup, history_finetune)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    print(f"[dlv3+] Building dataset from {images_dir} and {masks_dir}...")
    train_ds, val_ds = build_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_size=target_size,
        validation_split=validation_split,
    )

    # ── Phase 1: Warm-up (backbone frozen) ───────────────────────────────
    print("[dlv3+] Phase 1 — warm-up with frozen backbone...")
    model = build_deeplabv3plus(
        input_shape=(target_size[0], target_size[1], 3),
        n_classes=n_classes,
        freeze_backbone=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=n_classes, name='mean_iou'),
        ],
    )

    ckpt_warmup = os.path.join(output_dir, 'warmup.weights.h5')
    warmup_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_warmup, save_best_only=True, save_weights_only=True,
            monitor='val_loss', mode='min', verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1,
        ),
    ]

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_warmup,
        callbacks=warmup_callbacks,
    )

    # ── Phase 2: Fine-tune (full model) ──────────────────────────────────
    print("[dlv3+] Phase 2 — fine-tuning entire model...")

    # Unfreeze backbone
    backbone = model.get_layer('resnet50')
    backbone.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=n_classes, name='mean_iou'),
        ],
    )

    ckpt_finetune = os.path.join(output_dir, 'best_deeplabv3plus.weights.h5')
    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_finetune, save_best_only=True, save_weights_only=True,
            monitor='val_loss', mode='min', verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'), histogram_freq=0,
        ),
    ]

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        callbacks=finetune_callbacks,
    )

    model.summary()
    print("[dlv3+] Training complete!")
    return model, history_warmup, history_finetune

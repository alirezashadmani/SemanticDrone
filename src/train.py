import os
import tensorflow as tf
from .dataset import build_dataset
from .model import build_unet_model


def train_unet(
    images_dir,
    masks_dir,
    epochs=150,
    batch_size=8,
    buffer_size=500,
    target_size=(256, 256),
    n_classes=23,
    validation_split=0.2,
    output_dir='data/output/semantic_drone',
):
    """
    Train the U-Net semantic segmentation model.

    Args:
        images_dir: Path to directory containing original RGB images.
        masks_dir: Path to directory containing the segmentation masks.
        epochs: Max number of training epochs.
        batch_size: Batch size for tf.data.
        buffer_size: Shuffle buffer size.
        target_size: Target resolution (height, width).
        n_classes: Number of distinct segmentation classes.
        validation_split: Fraction of data held out for validation.
        output_dir: Directory for model checkpoints and logs.

    Returns:
        (model, history) tuple.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Build datasets ────────────────────────────────────────────────
    print(f"[train] Building dataset from {images_dir} and {masks_dir}...")
    train_ds, val_ds = build_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_size=target_size,
        validation_split=validation_split,
    )

    # ── 2. Build model ───────────────────────────────────────────────────
    print("[train] Building U-Net model...")
    input_shape = (target_size[0], target_size[1], 3)
    model = build_unet_model(input_shape=input_shape, n_classes=n_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=n_classes, name='mean_iou'),
        ],
    )
    model.summary()

    # ── 3. Callbacks ─────────────────────────────────────────────────────
    checkpoint_path = os.path.join(output_dir, 'best_unet.weights.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=0,
        ),
    ]

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("[train] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("[train] Training complete!")
    return model, history

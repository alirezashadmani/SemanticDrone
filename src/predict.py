"""
Inference and evaluation utilities for the Semantic Drone segmentation models.

Usage examples:
    # From a notebook / script:
    from src.predict import load_and_predict, evaluate_model, visualize_predictions

    model = tf.keras.models.load_model(...)      # or build + load weights
    evaluate_model(model, val_ds, n_classes=23)   # prints per-class IoU
    visualize_predictions(model, val_ds, num=4)   # side-by-side plots
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .dataset import CLASS_RGB


# ── Colour map for visualisation ─────────────────────────────────────────────

# Normalised [0-1] RGB palette aligned with class indices
_PALETTE = np.array(CLASS_RGB, dtype=np.float32) / 255.0

CLASS_NAMES = [
    'unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water', 'rocks',
    'pool', 'vegetation', 'roof', 'wall', 'window', 'door', 'fence',
    'fence-pole', 'person', 'dog', 'car', 'bicycle', 'tree', 'bald-tree',
    'ar-marker', 'obstacle',
]


def create_mask(pred_logits):
    """Convert model output logits → class-ID mask (H, W, 1)."""
    pred_mask = tf.argmax(pred_logits, axis=-1)  # (B, H, W)
    return pred_mask[..., tf.newaxis]             # (B, H, W, 1)


def mask_to_rgb(mask_2d):
    """Map a 2-D class-ID array → RGB image using the drone palette."""
    mask_np = np.asarray(mask_2d, dtype=np.int32)
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cid in range(len(_PALETTE)):
        rgb[mask_np == cid] = _PALETTE[cid]
    return rgb


# ── Single-image prediction ─────────────────────────────────────────────────

def load_and_predict(model, image_path, target_size=(256, 256)):
    """
    Load a single image from *image_path*, run inference, and return
    (input_image, predicted_mask) as numpy arrays.
    """
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size, method='nearest')

    logits = model.predict(img[tf.newaxis, ...])  # (1, H, W, C)
    pred = create_mask(logits)                     # (1, H, W, 1)

    return img.numpy(), pred[0].numpy().squeeze()


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, dataset, n_classes=23):
    """
    Compute overall and per-class IoU on *dataset*.

    Args:
        model: Compiled Keras model.
        dataset: ``tf.data.Dataset`` yielding (images, masks) batches.
        n_classes: Number of segmentation classes.

    Returns:
        dict with 'mean_iou' and 'per_class_iou'.
    """
    m = tf.keras.metrics.MeanIoU(num_classes=n_classes)

    # Accumulate confusion matrix over the full dataset
    for images, masks in dataset:
        logits = model.predict(images, verbose=0)
        preds = tf.argmax(logits, axis=-1)      # (B, H, W)
        labels = tf.squeeze(masks, axis=-1)      # (B, H, W)
        m.update_state(labels, preds)

    # Extract per-class IoU from the confusion matrix
    cm = m.total_cm.numpy()                      # (n_classes, n_classes)
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    denom = row_sum + col_sum - diag
    per_class = np.where(denom > 0, diag / denom, 0.0)

    mean_iou = float(np.nanmean(per_class[per_class > 0]))

    print(f"\n{'Class':<15}  IoU")
    print('-' * 28)
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_class)):
        print(f"{name:<15}  {iou:.4f}")
    print('-' * 28)
    print(f"{'Mean IoU':<15}  {mean_iou:.4f}\n")

    return {'mean_iou': mean_iou, 'per_class_iou': dict(zip(CLASS_NAMES, per_class.tolist()))}


# ── Visualisation ────────────────────────────────────────────────────────────

def visualize_predictions(model, dataset, num=4):
    """
    Plot *num* samples from *dataset* showing input, ground truth, and prediction.
    """
    fig, axes = plt.subplots(num, 3, figsize=(14, 4 * num))
    if num == 1:
        axes = axes[np.newaxis, :]

    titles = ['Input Image', 'Ground Truth', 'Prediction']
    for ax_row, title in zip(axes[0], titles):
        ax_row.set_title(title, fontsize=13)

    sample_idx = 0
    for images, masks in dataset:
        batch_logits = model.predict(images, verbose=0)
        batch_preds = create_mask(batch_logits)

        for i in range(images.shape[0]):
            if sample_idx >= num:
                break

            img_np = images[i].numpy()
            gt_np = masks[i].numpy().squeeze()
            pr_np = batch_preds[i].numpy().squeeze()

            axes[sample_idx, 0].imshow(img_np)
            axes[sample_idx, 1].imshow(mask_to_rgb(gt_np))
            axes[sample_idx, 2].imshow(mask_to_rgb(pr_np))

            for ax in axes[sample_idx]:
                ax.axis('off')

            sample_idx += 1

        if sample_idx >= num:
            break

    plt.tight_layout()
    plt.show()

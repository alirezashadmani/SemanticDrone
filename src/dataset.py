import os
import tensorflow as tf
from glob import glob


# ── Semantic Drone Dataset colour-palette → class-ID lookup ──────────────────
# Each RGB triplet maps to one of 23 classes.  Source: dataset documentation.
CLASS_RGB = [
    (0, 0, 0),        # 0  unlabeled
    (128, 64, 128),    # 1  paved-area
    (130, 76, 0),      # 2  dirt
    (0, 102, 0),       # 3  grass
    (112, 103, 87),    # 4  gravel
    (28, 42, 168),     # 5  water
    (48, 41, 30),      # 6  rocks
    (0, 50, 89),       # 7  pool
    (107, 142, 35),    # 8  vegetation
    (70, 70, 70),      # 9  roof
    (102, 102, 156),   # 10 wall
    (254, 228, 12),    # 11 window
    (254, 148, 12),    # 12 door
    (190, 153, 153),   # 13 fence
    (153, 153, 153),   # 14 fence-pole
    (255, 22, 96),     # 15 person
    (102, 51, 0),      # 16 dog
    (9, 143, 150),     # 17 car
    (119, 11, 32),     # 18 bicycle
    (51, 51, 0),       # 19 tree
    (190, 250, 190),   # 20 bald-tree
    (112, 150, 146),   # 21 ar-marker
    (2, 135, 115),     # 22 obstacle
]

# Build a TF lookup tensor: shape (256, 256, 256) would be huge, so we use a
# hash-based approach instead — encode each RGB as a single int32 key.
_rgb_arr = tf.constant(CLASS_RGB, dtype=tf.int32)                # (23, 3)
_keys = _rgb_arr[:, 0] * 65536 + _rgb_arr[:, 1] * 256 + _rgb_arr[:, 2]
_vals = tf.range(len(CLASS_RGB), dtype=tf.int32)
_class_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_keys, _vals), default_value=0
)


def _rgb_mask_to_class_ids(mask_rgb):
    """Convert an RGB mask (H, W, 3) uint8 → class-ID mask (H, W, 1) int32."""
    mask_int = tf.cast(mask_rgb, tf.int32)
    flat_keys = mask_int[:, :, 0] * 65536 + mask_int[:, :, 1] * 256 + mask_int[:, :, 2]
    class_ids = _class_table.lookup(flat_keys)
    return class_ids[..., tf.newaxis]  # (H, W, 1)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def process_path(image_path, mask_path):
    """Load and decode an image + mask pair from file paths."""
    # Image
    img_str = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_str, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Mask — decode as RGB then map to class IDs
    mask_str = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_str, channels=3)
    mask = _rgb_mask_to_class_ids(mask)

    return img, mask


def preprocess(image, mask, target_size=(256, 256)):
    """Resize image and mask to *target_size* using nearest-neighbour."""
    input_image = tf.image.resize(image, target_size, method='nearest')
    input_mask = tf.image.resize(mask, target_size, method='nearest')
    return input_image, input_mask


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(image, mask):
    """Apply random augmentations (spatial + colour jitter) to a single pair."""
    # Random horizontal flip — apply same decision to both
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Colour jitter (image only, not masks)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


# ── Dataset builder ──────────────────────────────────────────────────────────

def _match_by_stem(images_dir, masks_dir):
    """Match image↔mask pairs by filename stem, ignoring extension."""
    image_files = sorted(glob(os.path.join(images_dir, '*.*')))
    mask_files = sorted(glob(os.path.join(masks_dir, '*.*')))

    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    msk_map = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    # Filter out non-image hidden files (e.g. .DS_Store)
    img_map = {k: v for k, v in img_map.items() if not k.startswith('.')}
    msk_map = {k: v for k, v in msk_map.items() if not k.startswith('.')}

    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    if not common:
        raise ValueError(
            f"No matching image↔mask pairs found in\n  images: {images_dir}\n  masks:  {masks_dir}"
        )

    images = [img_map[k] for k in common]
    masks = [msk_map[k] for k in common]
    return images, masks


def build_dataset(
    images_dir,
    masks_dir,
    batch_size=8,
    buffer_size=500,
    target_size=(256, 256),
    validation_split=0.2,
    apply_augmentation=True,
):
    """
    Build train and validation ``tf.data.Dataset`` pipelines.

    Args:
        images_dir: Directory containing original RGB images.
        masks_dir: Directory containing segmentation masks.
        batch_size: Training batch size.
        buffer_size: Shuffle buffer size.
        target_size: (height, width) to resize images to.
        validation_split: Fraction of data reserved for validation.
        apply_augmentation: Whether to augment the training set.

    Returns:
        (train_dataset, val_dataset) — both batched and prefetched.
    """
    image_list, mask_list = _match_by_stem(images_dir, masks_dir)
    total = len(image_list)
    split_idx = int(total * (1 - validation_split))
    print(f"[dataset] {total} pairs found — {split_idx} train / {total - split_idx} val")

    def _make_pipeline(img_paths, msk_paths, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.constant(img_paths), tf.constant(msk_paths))
        )
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(
            lambda img, msk: preprocess(img, msk, target_size=target_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if is_training and apply_augmentation:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        if is_training:
            ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_pipeline(image_list[:split_idx], mask_list[:split_idx], is_training=True)
    val_ds = _make_pipeline(image_list[split_idx:], mask_list[split_idx:], is_training=False)
    return train_ds, val_ds

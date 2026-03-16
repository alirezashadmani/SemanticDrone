import os
import tensorflow as tf
from glob import glob

def process_path(image_path, mask_path):
    """Loads and decodes the image and its corresponding mask."""
    # Load and process image
    img_str = tf.io.read_file(image_path)
    # The original images are .jpg
    img = tf.image.decode_jpeg(img_str, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Load and process mask
    mask_str = tf.io.read_file(mask_path)
    # The masks are .png
    mask = tf.image.decode_png(mask_str, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    
    return img, mask

def preprocess(image, mask, target_size=(96, 128)):
    """Resizes the loaded images and masks to target dimensions."""
    input_image = tf.image.resize(image, target_size, method='nearest')
    input_mask = tf.image.resize(mask, target_size, method='nearest')
    return input_image, input_mask

def build_dataset(images_dir, masks_dir, batch_size=16, buffer_size=500, target_size=(96, 128)):
    """
    Builds a complete tf.data.Dataset pipeline.
    
    Args:
        images_dir: Path to directory containing original RGB images.
        masks_dir: Path to directory containing the segmentation masks.
        batch_size: The batch size to yield.
        buffer_size: Number of elements from which to randomly sample.
        target_size: Tuple (height, width) to resize images to.
        
    Returns:
        A batched and prefetched tf.data.Dataset ready for training.
    """
    # Grab sorted filenames so they align perfectly
    image_list = sorted(glob(os.path.join(images_dir, '*.*')))
    mask_list = sorted(glob(os.path.join(masks_dir, '*.*')))
    
    if not image_list or not mask_list:
        raise ValueError(f"Could not find images in {images_dir} or masks in {masks_dir}. Check paths!")
        
    images_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)
    
    # Initialize pipeline
    dataset = tf.data.Dataset.from_tensor_slices((images_filenames, masks_filenames))
    
    # Load & decode
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Resize
    dataset = dataset.map(
        lambda img, mask: preprocess(img, mask, target_size=target_size), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Shuffle, batch, & prefetch
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

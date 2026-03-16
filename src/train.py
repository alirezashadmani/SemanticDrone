import os
import tensorflow as tf
from .dataset import build_dataset
from .model import build_unet_model

def train_unet(
    images_dir,
    masks_dir,
    epochs=150,
    batch_size=16,
    buffer_size=500,
    target_size=(96, 128),
    n_classes=23,
    output_dir='data/output/semantic_drone'
):
    """
    Orchestrates the training for the Semantic Drone U-Net Model.
    
    Args:
        images_dir: Path to directory containing original RGB images.
        masks_dir: Path to directory containing the segmentation masks.
        epochs: Max number of training epochs.
        batch_size: Batch size to use for tf.data.
        buffer_size: Buffer size for dataset shuffling.
        target_size: Target resolution (height, width).
        n_classes: Number of distinct segmentation classes.
        output_dir: Directory to save model checkpoints and logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build Dataset
    print(f"Building dataset from {images_dir} and {masks_dir}...")
    dataset = build_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_size=target_size
    )
    
    # Since we don't have an explicit validation set split defined in the 
    # original notebook, we'll train on the whole dataset. 
    # (A professional pipeline usually splits this, but we'll stick to original logic).
    train_dataset = dataset
    
    # 2. Build Model
    print("Building U-Net Model...")
    input_shape = (target_size[0], target_size[1], 3)
    model = build_unet_model(input_shape=input_shape, n_classes=n_classes)
    
    # Compile
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # 3. Setup Callbacks
    checkpoint_filepath = os.path.join(output_dir, 'best_unet_model.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=False, # Save all epochs since we don't have validation monitor right now
            save_weights_only=True,
            monitor='loss',
            mode='min',
            verbose=1
        ),
        # Stop training if the training loss stops improving for 10 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 4. Train
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Training complete!")
    return model, history

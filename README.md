# Semantic Drone Segmentation

This repository contains a Convolutional Neural Network (U-Net) designed for semantic segmentation of high-resolution drone imagery. 

The original codebase was refactored from a monolithic Jupyter Notebook into a structured, scalable, and memory-efficient Python package.

## Refactoring Highlights
1. **Out-of-Memory (OOM) Prevention:** Replaced the legacy loop-based data loading (which loaded 400x 4000x6000 images directly into RAM) with a scalable `tf.data.Dataset` pipeline. Images and masks are now dynamically loaded, resized, and streamed batch-by-batch to the GPU.
2. **Environment Independence:** Removed hardcoded Google Colab paths (`/content/drive/...`).
3. **Modular Architecture:** Extracted core logic into standard Python modules (`src/dataset.py`, `src/model.py`, `src/train.py`).
4. **Professional Callbacks:** Added Keras `EarlyStopping` and `ModelCheckpoint` callbacks to ensure the best model weights are saved automatically.

## Repository Structure
- `src/dataset.py`: Contains the `tf.data.Dataset` pipeline for loading, decoding, and resizing images and masks.
- `src/model.py`: Defines the U-Net architecture.
- `src/train.py`: Orchestrates the training process, compiles the model, and configures callbacks.
- `run_pipeline.ipynb`: The main entry-point notebook. Configure your local data paths here and execute the training pipeline.
- `Image_Segmentation_U_net.ipynb`: The original legacy notebook (kept for reference).

## Usage

### 1. Extract the Dataset
Download and extract the semantic drone dataset (`semantic_drone_dataset-...zip`) into a local directory, for example:
```
data/
  ├── original_images/
  └── label_images/
```

*(Note: The raw datasets are ignored via `.gitignore` to prevent exceeding GitHub's 100 MB file limit).*

### 2. Configure Paths
Open `run_pipeline.ipynb` and update the `IMAGES_DIR` and `MASKS_DIR` variables to point to your extracted dataset folders.

### 3. Run the Pipeline
Run the cells in `run_pipeline.ipynb`. The pipeline will automatically:
- Construct the `tf.data.Dataset`
- Build the U-Net model
- Start training, saving the best weights to the `OUTPUT_DIR` (default: `data/output/semantic_drone/`).

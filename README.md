# Semantic Drone Segmentation

Semantic segmentation of high-resolution drone imagery (400 images, 23 classes) using two architectures:

| Model | Backbone | Key feature |
|-------|----------|-------------|
| **U-Net** (improved) | From scratch | BatchNorm, broad dropout, lightweight |
| **DeepLabV3+** | ResNet50 (ImageNet) | ASPP multi-scale context, two-phase training |

## Highlights

- **Proper mask decoding** — RGB colour-palette → class-ID via hash-table lookup (23 Semantic Drone classes).
- **80/20 train/val split** with all callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`) monitoring `val_loss`.
- **Data augmentation** — random flips, brightness / contrast / saturation jitter on the training set.
- **BatchNormalization** in every encoder and decoder conv block.
- **Mean IoU** tracked during training and per-class IoU evaluation at inference time.
- **DeepLabV3+** with Atrous Spatial Pyramid Pooling and a two-phase training strategy (frozen backbone warm-up → full fine-tune).
- **Scalable `tf.data.Dataset` pipeline** — images are streamed, resized, augmented, and prefetched batch-by-batch.
- **Inference & visualisation module** — single-image prediction, per-class IoU report, and colour-mapped side-by-side plots.

## Repository Structure

```
src/
  dataset.py              # tf.data pipeline: loading, mask decoding, augmentation, train/val split
  model.py                # Improved U-Net with BatchNorm and graduated dropout
  deeplabv3plus.py        # DeepLabV3+ with ResNet50 backbone and ASPP
  train.py                # U-Net training orchestration
  train_deeplabv3plus.py  # DeepLabV3+ two-phase training (warm-up → fine-tune)
  predict.py              # Inference, per-class IoU evaluation, visualisation
  __init__.py             # Public API re-exports
run_pipeline.ipynb        # Main notebook — trains both models, evaluates, and compares
```

## Usage

### 1. Extract the Dataset

Download the [Semantic Drone Dataset](https://www.tugraz.at/index.php?id=22387) and extract it:

```
data/
  ├── original_images/    # 400 high-res RGB .jpg files
  └── label_images/       # Corresponding .png segmentation masks
```

> Raw data and model outputs are git-ignored to stay under GitHub's file-size limit.

### 2. Configure Paths

Open `run_pipeline.ipynb` and update `IMAGES_DIR` / `MASKS_DIR` if your layout differs from the default.

### 3. Run the Pipeline

Execute the notebook cells in order. The pipeline will:

1. Build the `tf.data.Dataset` with augmentation and an 80/20 split.
2. Train the **improved U-Net** (BatchNorm, Mean IoU, ReduceLROnPlateau).
3. Evaluate U-Net — per-class IoU table and prediction visualisations.
4. Train **DeepLabV3+** in two phases (frozen backbone → full fine-tune).
5. Evaluate DeepLabV3+ and produce a side-by-side comparison chart.

Best weights are saved to `data/output/unet/` and `data/output/deeplabv3plus/`.

### Using Individual Modules

```python
from src.deeplabv3plus import build_deeplabv3plus
from src.predict import load_and_predict, evaluate_model, visualize_predictions

model = build_deeplabv3plus(input_shape=(256, 256, 3), n_classes=23)
model.load_weights('data/output/deeplabv3plus/best_deeplabv3plus.weights.h5')

img, pred_mask = load_and_predict(model, 'path/to/image.jpg')
```

## Class Palette (23 classes)

| ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|
| 0 | unlabeled | 8 | vegetation | 16 | dog |
| 1 | paved-area | 9 | roof | 17 | car |
| 2 | dirt | 10 | wall | 18 | bicycle |
| 3 | grass | 11 | window | 19 | tree |
| 4 | gravel | 12 | door | 20 | bald-tree |
| 5 | water | 13 | fence | 21 | ar-marker |
| 6 | rocks | 14 | fence-pole | 22 | obstacle |
| 7 | pool | 15 | person | | |

# Data Preprocessing

## Raw data → class folders

The source data is the Severstal steel-defect Kaggle dataset: a flat folder
of images plus a `train.csv` with `ImageId, ClassId, EncodedPixels` rows
(one row per defect instance; `EncodedPixels` is a segmentation mask, unused
here since this is a classification model). `steel_defect/organize_dataset.py`
converts that into the class-per-folder layout the rest of the pipeline
expects:

```
data/steel_defect/
├── no_defect/
├── defect_1/
├── defect_2/
├── defect_3/
└── defect_4/
```

Rules it applies while sorting:

- An image with **no row** in `train.csv` → `no_defect`.
- An image with exactly **one** distinct `ClassId` → that class's folder.
- An image with **more than one** distinct `ClassId` (multiple defect types
  on the same image) → **skipped entirely**, rather than force-assigned to
  a single class. This avoids injecting label noise into a single-label
  classifier.
- Files are **copied**, not moved, by default (`--move` opts into moving).
- `--dry-run` prints the resulting counts without touching disk — useful for
  sanity-checking the split before committing to it.

```bash
python organize_dataset.py \
    --images-dir /path/to/train_images \
    --csv /path/to/train.csv \
    --output-dir data/steel_defect \
    --dry-run
```

## Scanning the dataset (`build_file_list`)

`steel_defect/dataset.py::build_file_list()` walks `data/steel_defect/`,
keeps only subfolders whose name is in `CLASS_NAMES`
(`["no_defect", "defect_1", "defect_2", "defect_3", "defect_4"]`), and
collects every file whose suffix is in `IMAGE_EXTENSIONS`
(`.png, .jpg, .jpeg, .bmp, .tiff`) into a sorted list of
`(image_path_str, label_index)` tuples. Label indices come from
`CLASS_NAMES.index(folder.name)`, so `no_defect` is always `0` and
`defect_1..4` are `1..4`.

Sorting the final list (`results.sort(key=lambda pair: pair[0])`) matters for
reproducibility — it guarantees the same input ordering across runs, which
in turn makes the stratified splits below deterministic given a fixed seed.

## Train / val / test split (`create_splits`)

Splitting is done with two calls to
`sklearn.model_selection.train_test_split`, both **stratified by label** so
each class keeps its proportional representation in every split:

1. **First split** — pulls out the test set: `test_size = 1 - train_ratio -
   val_ratio` (default `1 - 0.7 - 0.15 = 0.15`), stratified on the full
   label list.
2. **Second split** — splits the remaining train+val portion using an
   **adjusted** validation ratio, `val_ratio / (train_ratio + val_ratio)`
   (default `0.15 / 0.85 ≈ 0.176`), stratified on the labels of that
   remaining subset (not the full dataset — using the full-dataset label
   list here would silently reintroduce samples that were already routed to
   test).

Default ratios: 70% train / 15% val / 15% test. A real run against the
Severstal-derived dataset produced:

| Split | Samples |
|---|---|
| Train | 8,497 |
| Val | 1,822 |
| Test | 1,822 |
| **Total** | **12,141** |

`seed=42` by default, giving fully reproducible splits across runs.

## Image loading and augmentation

### `SteelDataset.__getitem__` (DATA-2)

For each index: read the image with `cv2.imread` (raising `FileNotFoundError`
if it comes back `None`, e.g. a corrupt or missing file), convert
`BGR → RGB` with `cv2.cvtColor`, then apply the active Albumentations
transform:

```python
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if self.transform is not None:
    result = self.transform(image=image)
    image = result["image"]
return image, label
```

### Training transforms (`build_train_transforms`, PREPROCESS-1)

```python
A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

Augmentation is deliberately light: a horizontal flip (steel surface defects
have no inherent "up" orientation that flipping would break) and mild
brightness/contrast jitter (robustness to lighting variation across camera
setups), each applied with modest probability so the model still sees a
majority of clean, undistorted samples.

### Validation/test transforms (`build_val_transforms`, PREPROCESS-2)

```python
A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

Deterministic — no augmentation — so validation and test metrics measure the
model's actual generalization rather than variance introduced by random
transforms. Both pipelines use ImageNet normalization stats
(`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`), which keeps
input statistics in the range most conv-net initializations and batchnorm
layers are tuned for, even though this model is trained from scratch rather
than fine-tuned from an ImageNet checkpoint.

## DataLoaders

`create_dataloaders()` wraps the train/val splits in `SteelDataset` +
`DataLoader`: training shuffles every epoch (`shuffle=True`), validation
doesn't (`shuffle=False`, since order doesn't matter and consistency aids
debugging), both use `pin_memory=True` for faster host→GPU transfer.
# Project Overview

## Problem

Manual visual inspection for surface defects on steel sheets is slow and
inconsistent between inspectors. This project builds an automated
classifier that looks at a steel surface image and sorts it into one of
five categories, so defective sheets can be flagged for review without a
human checking every image by hand.

## Dataset

Images are sourced from the Severstal steel-defect dataset (Kaggle): a flat
folder of images plus a `train.csv` mapping image IDs to defect class IDs
and segmentation masks. `organize_dataset.py` converts that into a
class-per-folder layout — images with no listed defect become `no_defect`,
images with exactly one defect type go into the matching folder
(`defect_1`–`defect_4`), and images labeled with more than one defect type
are skipped rather than force-assigned to a single class. From there,
`build_file_list()` + `create_splits()` produce a stratified 70/15/15
train/val/test split (8,497 / 1,822 / 1,822 images on the run documented in
[Training](training.md)), so each class is proportionally represented in
every split.

## Approach

A CNN trained from scratch — `SteelCNN`, ~113K parameters — rather than a
fine-tuned pretrained backbone, since the goal was to build and understand
every stage of the pipeline directly. The architecture uses three
convolutional blocks, each with a residual (skip) connection via a 1×1
projection shortcut, followed by global average pooling and a small fully
connected classifier head. See [Architecture](model-architecture.md) for
the full design and [Data Preprocessing](data-preprocessing.md) for the
transform pipeline (resize, light augmentation, ImageNet normalization).

Training uses `CrossEntropyLoss` + Adam, with checkpointing on validation
accuracy so the saved model reflects genuine generalization rather than
the epoch with the lowest training loss. The trained model is served
through a Streamlit app (`app.py`) that classifies uploaded or browsed
images and overlays a Grad-CAM heatmap showing which regions of the image
drove the prediction — see [Results](results.md) for how it performed and
[Challenges](challenges.md) for what broke along the way.
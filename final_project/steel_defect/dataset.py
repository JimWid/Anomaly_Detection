"""
Dataset module for steel defect classification.

Provides functions to scan the class-directory dataset structure,
create train/val/test splits, and a PyTorch Dataset class for
loading and transforming images.
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from steel_defect.utils import setup_logging, CLASS_NAMES, DATA_DIR, IMAGE_SIZE

logger = setup_logging(__name__)

# File extensions to include when scanning for images
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def build_file_list(data_dir: Path | str | None = None) -> list[tuple[str, int]]:
    """
    DATA-1: Scan the dataset directory and build a list of (path, label) pairs.

    The dataset is organized as one subdirectory per class:
        data/steel_defect/
        ├── no_defect/   → label 0
        ├── defect_1/    → label 1
        ├── defect_2/    → label 2
        ├── defect_3/    → label 3
        └── defect_4/    → label 4

    For each class folder that exists in CLASS_NAMES:
        1. Get the label index from CLASS_NAMES.index(folder_name)
        2. Collect all image files (check IMAGE_EXTENSIONS for valid suffixes)
        3. Append (str(image_path), label_index) to the result list

    Args:
        data_dir: Root directory containing class subdirectories.
                  Defaults to DATA_DIR from utils.

    Returns:
        List of (image_path_str, label_index) tuples, sorted by path.

    Raises:
        FileNotFoundError: If data_dir does not exist.

    Hint:
        - Use Path(data_dir).iterdir() to loop over subdirectories
        - Check if folder.name is in CLASS_NAMES
        - Use path.suffix.lower() to check file extensions
        - Remember to sort the final list for reproducibility
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Place your images in subdirectories: {', '.join(CLASS_NAMES)}"
        )

    # ┌──────────────────────────────────────────────┐
    # │  DATA-1: Write your code below               │
    # └──────────────────────────────────────────────┘

    results = []
    # MAX_PER_CLASS = 1000

    for folder in data_dir.iterdir():
        if not folder.is_dir() or folder.name not in CLASS_NAMES:
            continue

        label = CLASS_NAMES.index(folder.name)

        for image_path in folder.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                results.append((str(image_path), label))

    results.sort(key=lambda pair: pair[0])

    #    folder_images = [ 
    #        image_path for image_path in folder.iterdir() if
    #        image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    #        ]
    #
    #    folder_images.sort()
    #
    #    for image_path in folder_images[:MAX_PER_CLASS]:        
    #            results.append((str(image_path), label))

    return results


def create_splits(
    file_list: list[tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    DATA-3: Split the file list into train, validation, and test sets.

    Use sklearn.model_selection.train_test_split to:
        1. First split: separate test set (remaining ratio = 1 - train - val)
        2. Second split: separate train and val from the non-test portion

    Both splits should use stratification (stratify by labels) to ensure
    each class is proportionally represented in all splits.

    Args:
        file_list: List of (image_path, label) tuples from build_file_list().
        train_ratio: Fraction of data for training (default 0.7).
        val_ratio: Fraction of data for validation (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_list, val_list, test_list), each a list of
        (image_path, label) tuples.

    Hint:
        - Extract labels: [label for _, label in file_list]
        - test_ratio = 1.0 - train_ratio - val_ratio
        - First split: train_test_split(file_list, test_size=test_ratio,
                                         stratify=labels, random_state=seed)
        - val_ratio relative to the remaining = val_ratio / (train_ratio + val_ratio)
        - Second split on the non-test portion using the adjusted ratio
    """
    # ┌──────────────────────────────────────────────┐
    # │  DATA-3: Write your code below               │
    # └──────────────────────────────────────────────┘

    labels = [label for _, label in file_list]
    test_ratio = 1.0 - train_ratio - val_ratio

    # train + val split and test split
    train_val_split, test_split = train_test_split(file_list, test_size=test_ratio, stratify=labels, random_state=seed)

    # labels and new val ratio
    train_val_labels = [label for _, label in train_val_split]
    new_val_ratio = val_ratio / (train_ratio + val_ratio)

    # train split and val split
    train_split, val_split = train_test_split(train_val_split, test_size=new_val_ratio, stratify=train_val_labels, random_state=seed)

    return (train_split, val_split, test_split)


class SteelDataset(Dataset):
    """
    PyTorch Dataset for steel defect images.

    Loads images from disk, applies transforms, and returns
    (tensor, label) pairs ready for the DataLoader.
    """

    def __init__(
        self,
        file_list: list[tuple[str, int]],
        transform: A.Compose | None = None,
    ):
        """
        Args:
            file_list: List of (image_path, label) tuples.
            transform: Albumentations transform pipeline to apply.
        """
        self.file_list = file_list
        self.transform = transform

        logger.info(
            "SteelDataset created | samples=%d | transform=%s",
            len(self.file_list),
            "yes" if transform else "none",
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
        DATA-2: Load an image and return (tensor, label).

        Steps:
            1. Get the (image_path, label) tuple at position idx
            2. Read the image with cv2.imread()
            3. Convert from BGR to RGB with cv2.cvtColor()
            4. If self.transform is not None, apply it:
               result = self.transform(image=image)
               image = result["image"]
            5. Return (image, label)

        Args:
            idx: Index into self.file_list.

        Returns:
            Tuple of (image_tensor, label_int).
            - image_tensor: (3, H, W) float32 tensor if transforms include ToTensorV2
            - label_int: integer class label

        Hint:
            - cv2.imread returns None if the file can't be read — check for this
            - cv2.cvtColor(img, cv2.COLOR_BGR2RGB) converts BGR to RGB
        """
        # ┌──────────────────────────────────────────────┐
        # │  DATA-2: Write your code below               │
        # └──────────────────────────────────────────────┘

        img_path, label = self.file_list[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            result = self.transform(image=image)
            image = result["image"]

        return (image, label) 


# ── Scaffold — DataLoader helper ──────────────────────────────

def create_dataloaders(
    train_list: list,
    val_list: list,
    train_transform: A.Compose,
    val_transform: A.Compose,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_list: Training (path, label) pairs.
        val_list: Validation (path, label) pairs.
        train_transform: Augmentation pipeline for training.
        val_transform: Deterministic pipeline for validation.
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_ds = SteelDataset(train_list, transform=train_transform)
    val_ds = SteelDataset(val_list, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders created | train=%d batches | val=%d batches | batch_size=%d",
        len(train_loader),
        len(val_loader),
        batch_size,
    )

    return train_loader, val_loader

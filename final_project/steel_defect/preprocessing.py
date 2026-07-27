"""
Image preprocessing pipeline.

Defines the transform pipelines used for training and validation/test.
Training transforms include data augmentation; validation transforms
are deterministic (resize + normalize only).
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from steel_defect.utils import setup_logging, IMAGE_SIZE

logger = setup_logging(__name__)

# ImageNet normalization constants (used by most pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms() -> A.Compose:
    """
    PREPROCESS-1: Build the training image transform pipeline.

    Creates an albumentations.Compose pipeline that applies these
    transforms in order:
        1. Resize to IMAGE_SIZE (256, 256)
        2. Horizontal flip with probability 0.5
        3. Random brightness/contrast with probability 0.3
        4. Normalize using IMAGENET_MEAN and IMAGENET_STD
        5. Convert to PyTorch tensor with ToTensorV2()

    Returns:
        A.Compose: The training transform pipeline.
    """

    # ┌──────────────────────────────────────────────┐
    # │  PREPROCESS-1: Write your code below         │
    # └──────────────────────────────────────────────┘

    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def build_val_transforms() -> A.Compose:
    """
    PREPROCESS-2: Build the validation/test image transform pipeline.

    This pipeline should be deterministic (no random augmentation).
    Applies these transforms in order:
        1. Resize to IMAGE_SIZE (256, 256)
        2. Normalize using IMAGENET_MEAN and IMAGENET_STD
        3. Convert to PyTorch tensor with ToTensorV2()

    Returns:
        A.Compose: The validation/test transform pipeline.
    """

    # ┌──────────────────────────────────────────────┐
    # │  PREPROCESS-2: Write your code below         │
    # └──────────────────────────────────────────────┘

    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])



# ── Scaffold — provided for Grad-CAM overlay ──────────────────
def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: RGB image (H, W, 3), uint8.
        heatmap: Grad-CAM map (H, W), float32 in [0, 1].
        alpha: Blend factor for the overlay.
        colormap: OpenCV colormap constant.

    Returns:
        Blended RGB image (H, W, 3), uint8.
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    logger.debug("Grad-CAM overlay applied | alpha=%.2f | size=(%d, %d)", alpha, w, h)
    return blended

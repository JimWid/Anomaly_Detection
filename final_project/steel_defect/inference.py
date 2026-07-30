"""
Inference module for steel defect classification.

Loads a trained CNN checkpoint and provides a predict() interface
for single-image classification with confidence scores.
"""
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from steel_defect.utils import (
    setup_logging,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    NUM_CLASSES,
    DEVICE,
)
from steel_defect.model import SteelCNN
from steel_defect.preprocessing import build_val_transforms

logger = setup_logging(__name__)


class SteelPredictor:
    """
    Steel defect classifier for inference.

    Loads a trained SteelCNN checkpoint and classifies individual
    images into defect categories.

    Usage:
        predictor = SteelPredictor()
        predictor.load_model()
        result = predictor.predict(image)
        print(result["label"])       # "defect_1"
        print(result["confidence"])  # 0.93
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | None = None,
    ):
        """
        Args:
            checkpoint_path: Path to the saved .pt checkpoint.
                             Defaults to CHECKPOINT_PATH from utils.
            device: Torch device. Defaults to DEVICE from utils.
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        self.device = device or DEVICE
        self.model: Optional[SteelCNN] = None
        self.transform = build_val_transforms()
        self._inference_count = 0

        logger.info(
            "SteelPredictor initialized | checkpoint=%s | device=%s",
            self.checkpoint_path,
            self.device,
        )

    def load_model(self) -> None:
        """
        INFER-1: Load the trained model from a checkpoint file.

        Steps:
            1. Check that self.checkpoint_path exists, raise FileNotFoundError if not
            2. Load the checkpoint dict: torch.load(self.checkpoint_path, map_location=self.device)
            3. Create a new SteelCNN instance: SteelCNN(num_classes=NUM_CLASSES)
            4. Load the saved weights: model.load_state_dict(checkpoint["model_state_dict"])
            5. Move to device: model.to(self.device)
            6. Set to eval mode: model.eval()
            7. Assign to self.model

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        Hint:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            The checkpoint dict has keys: "model_state_dict", "optimizer_state_dict",
            "epoch", "best_val_acc", "num_classes"
        """
        
        start = time.perf_counter()

        # ┌──────────────────────────────────────────────┐
        # │  INFER-1: Write your code below              │
        # └──────────────────────────────────────────────┘
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model = SteelCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Model loaded | time=%.0fms", elapsed_ms)

    def predict(self, image: np.ndarray) -> dict:
        """
        INFER-2: Run classification on a single image.

        Steps:
            1. If self.model is None, call self.load_model()
            2. Apply self.transform to the image:
               result = self.transform(image=image)
               tensor = result["image"]
            3. Add batch dimension: tensor = tensor.unsqueeze(0)
            4. Move to device: tensor = tensor.to(self.device)
            5. Forward pass with no gradients:
               with torch.no_grad():
                   logits = self.model(tensor)
            6. Convert logits to probabilities: probs = F.softmax(logits, dim=1)
            7. Get predicted class and confidence:
               confidence, predicted = probs.max(dim=1)
            8. Map predicted index to class name: CLASS_NAMES[predicted.item()]

        Args:
            image: RGB numpy array (H, W, 3), uint8.

        Returns:
            Dictionary with:
                - "label": str — predicted class name (e.g., "defect_2")
                - "confidence": float — confidence score (0-1)
                - "class_scores": dict — {class_name: probability} for all classes
                - "predicted_idx": int — predicted class index
                - "latency_ms": float — inference time in milliseconds

        Hint:
            probs_np = probs.squeeze().cpu().numpy()
            class_scores = {name: float(p) for name, p in zip(CLASS_NAMES, probs_np)}
        """
        if self.model is None:
            self.load_model()

        start = time.perf_counter()
        # ┌──────────────────────────────────────────────┐
        # │  INFER-2: Write your code below              │
        # └──────────────────────────────────────────────┘
        
        result = self.transform(image=image)
        tensor = result["image"]
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

        probs_np = probs.squeeze().cpu().numpy()
        class_scores = {name: float(p) for name, p in zip(CLASS_NAMES, probs_np)}

        predicted_idx = predicted.item()
        label = CLASS_NAMES[predicted_idx]
        confidence_val = confidence.item()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._inference_count += 1

        logger.info(
            "Inference #%d | label=%s | confidence=%.3f | time=%.1fms",
            self._inference_count, label, confidence_val, elapsed_ms,
        )

        return {
            "label": label,
            "confidence": confidence_val,
            "class_scores": class_scores,
            "predicted_idx": predicted_idx,
            "latency_ms": elapsed_ms,
        }

    def predict_from_file(self, image_path: str | Path) -> dict:
        """Convenience method to predict from a file path."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.predict(image)
        result["source_path"] = str(image_path)
        return result

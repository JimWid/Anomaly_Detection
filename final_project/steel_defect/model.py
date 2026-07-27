"""
CNN model definition for steel defect classification.

Students define a simple convolutional neural network with 3 conv blocks
followed by fully connected layers. The architecture is intentionally
simple — a good starting point before exploring deeper models like ResNet.
"""
import torch
import torch.nn as nn

from steel_defect.utils import NUM_CLASSES, setup_logging

logger = setup_logging(__name__)


class SteelCNN(nn.Module):
    """
    Simple CNN for steel defect classification.

    Architecture overview (what you will implement):
        Block 1: Conv2d(3→32)  → BatchNorm → ReLU → MaxPool(2)
        Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool(2)
        Block 3: Conv2d(64→128)→ BatchNorm → ReLU → MaxPool(2)
        AdaptiveAvgPool2d(1)  → Flatten
        FC: 128 → 64 → NUM_CLASSES

    Input:  (batch, 3, 256, 256)
    Output: (batch, NUM_CLASSES) — raw logits (no softmax)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        """
        MODEL-1: Define the CNN layers.

        Create the following layers as instance attributes:

        self.features — nn.Sequential containing 3 convolutional blocks:
            Block 1: nn.Conv2d(3, 32, kernel_size=3, padding=1)
                      nn.BatchNorm2d(32)
                      nn.ReLU()
                      nn.MaxPool2d(2)
            Block 2: nn.Conv2d(32, 64, kernel_size=3, padding=1)
                      nn.BatchNorm2d(64)
                      nn.ReLU()
                      nn.MaxPool2d(2)
            Block 3: nn.Conv2d(64, 128, kernel_size=3, padding=1)
                      nn.BatchNorm2d(128)
                      nn.ReLU()
                      nn.MaxPool2d(2)

        self.pool — nn.AdaptiveAvgPool2d(1)
            Reduces any spatial size to 1×1.

        self.classifier — nn.Sequential containing:
            nn.Flatten()
            nn.Linear(128, 64)
            nn.ReLU()
            nn.Dropout(0.3)
            nn.Linear(64, num_classes)

        Hint:
            super().__init__()    # MUST call this first!
            self.features = nn.Sequential(...)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(...)
        """
        # ┌──────────────────────────────────────────────┐
        # │  MODEL-1: Write your code below              │
        # └──────────────────────────────────────────────┘

        super().__init__()
        
        self.residual_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.shortcut_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=2),
            nn.BatchNorm2d(32)
        )

        self.residual_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.shortcut_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64)
        )

        self.residual_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.shortcut_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MODEL-2: Define the forward pass.

        Pass the input tensor through:
            1. self.features  — convolutional blocks
            2. self.pool      — adaptive average pooling
            3. self.classifier — flatten + fully connected layers

        Args:
            x: Input tensor of shape (batch, 3, 256, 256).

        Returns:
            Logits tensor of shape (batch, NUM_CLASSES).
            Do NOT apply softmax here — CrossEntropyLoss expects raw logits.

        Hint:
            x = self.features(x)
            x = self.pool(x)
            x = self.classifier(x)
            return x
        """
        # ┌──────────────────────────────────────────────┐
        # │  MODEL-2: Write your code below              │
        # └──────────────────────────────────────────────┘

        # Residual 1
        identity = x
        x1 = self.residual_block_1(x) + self.shortcut_1(identity)

        # Residual 2
        identity = x1
        x2 = self.residual_block_2(x1) + self.shortcut_2(identity)

        # Residual 3
        identity = x2
        x3 = self.residual_block_3(x2) + self.shortcut_3(identity)

        # Avg Pooling
        x = self.pool(x3)

        # Classifier to number of classes
        output = self.classifier(x)

        return output

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"SteelCNN(params={self.num_parameters:,})"

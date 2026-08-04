# Architecture

## Overview

`SteelCNN` (`steel_defect/model.py`) is a small residual convolutional network
for 5-class steel surface defect classification. It departs from the plain
3-block CNN described in the original homework spec: each block is paired
with a 1×1 projection shortcut, so the network learns residual corrections
rather than a single unbroken chain of convolutions.

```
Input (batch, 3, 256, 256)
        │
        ├────────────────────────────┐
        ▼                            ▼
  residual_block_1              shortcut_1
  Conv(3→32,k3,p1)               Conv(3→32,k1,s2)
  BN → ReLU → MaxPool(2)         BN
        │                            │
        └──────────► + ◄─────────────┘
                      │
                (batch, 32, 128, 128)
        │
        ├────────────────────────────┐
        ▼                            ▼
  residual_block_2              shortcut_2
  Conv(32→64,k3,p1)              Conv(32→64,k1,s2)
  BN → ReLU → MaxPool(2)         BN
        │                            │
        └──────────► + ◄─────────────┘
                      │
                (batch, 64, 64, 64)
        │
        ├────────────────────────────┐
        ▼                            ▼
  residual_block_3              shortcut_3
  Conv(64→128,k3,p1)             Conv(64→128,k1,s2)
  BN → ReLU → MaxPool(2)         BN
        │                            │
        └──────────► + ◄─────────────┘
                      │
                (batch, 128, 32, 32)
                      │
                      ▼
          AdaptiveAvgPool2d(1) → (batch, 128, 1, 1)
                      │
                      ▼
                 classifier
             Flatten → (batch, 128)
             Linear(128→64) → ReLU → Dropout(0.3)
             Linear(64→5)
                      │
                      ▼
          Output: (batch, 5) raw logits
```

## Why a shortcut per block

A residual addition only works if the two tensors being summed have the same
shape. Each `residual_block_N` changes both the channel count (via its 3×3
conv) and the spatial size (via `MaxPool2d(2)`), so the identity path can't
be added back in unmodified — it has to be projected into the same shape
first. Each `shortcut_N` does that with a single 1×1 convolution using
`stride=2` (matching the block's downsampling) and an output channel count
matching the block's output channels, followed by `BatchNorm2d` for
consistency with the main path.

This keeps the connections **local to each block** — every block adds its
own shortcut of its own input, rather than trying to connect every block
back to the raw network input (which would need mismatched channel/size fixes
at every stage and defeats the point of a block-local residual design).

## Layer-by-layer

| Stage | Main path | Shortcut | Output shape |
|---|---|---|---|
| Block 1 | Conv2d(3,32,k=3,p=1) → BN → ReLU → MaxPool(2) | Conv2d(3,32,k=1,s=2) → BN | (32, 128, 128) |
| Block 2 | Conv2d(32,64,k=3,p=1) → BN → ReLU → MaxPool(2) | Conv2d(32,64,k=1,s=2) → BN | (64, 64, 64) |
| Block 3 | Conv2d(64,128,k=3,p=1) → BN → ReLU → MaxPool(2) | Conv2d(64,128,k=1,s=2) → BN | (128, 32, 32) |
| Pool | AdaptiveAvgPool2d(1) | — | (128, 1, 1) |
| Classifier | Flatten → Linear(128,64) → ReLU → Dropout(0.3) → Linear(64,5) | — | (5,) logits |

The model has **113,285 trainable parameters** (`model.num_parameters`),
confirmed from an actual run: `SteelCNN(params=113,285)`.

## Forward pass

```python
def forward(self, x):
    identity = x
    x1 = self.residual_block_1(x) + self.shortcut_1(identity)

    identity = x1
    x2 = self.residual_block_2(x1) + self.shortcut_2(identity)

    identity = x2
    x3 = self.residual_block_3(x2) + self.shortcut_3(identity)

    x = self.pool(x3)
    return self.classifier(x)
```

No softmax is applied — `nn.CrossEntropyLoss` (used in training, see
[Training](training.md)) expects raw logits and applies `LogSoftmax`
internally. At inference time, `SteelPredictor.predict()` applies
`F.softmax` explicitly to get class probabilities.

## Grad-CAM compatibility

Because there's no single `self.features` sequential block, Grad-CAM
(`steel_defect/gradcam.py`) hooks the last `Conv2d` inside
`residual_block_3` rather than a generic `features[-1]`:

```python
for layer in reversed(list(predictor.model.residual_block_3.children())):
    if isinstance(layer, torch.nn.Conv2d):
        target_layer = layer
        break
```

`residual_block_3` — not `shortcut_3` — is used because it's the deepest
layer on the main representational path, right before pooling and
classification; the shortcut is only a shape-matching projection, not where
the network's learned features live.
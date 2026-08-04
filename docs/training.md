# Training

## Pipeline overview

`steel_defect/train.py::train()` orchestrates the full run: build the file
list → stratified split → transforms → dataloaders → model → loss/optimizer
→ epoch loop with checkpointing on validation improvement.

```bash
python -m steel_defect.train
python -m steel_defect.train --epochs 30 --batch_size 64 --lr 0.0005
```

## Loss and optimizer (`setup_training`, TRAIN-1)

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

`CrossEntropyLoss` combines `LogSoftmax` + `NLLLoss` and expects raw logits
(shape `(batch, 5)`) plus integer class targets (shape `(batch,)`) — matching
`SteelCNN.forward()`'s output exactly, since the model intentionally doesn't
apply softmax internally. Adam is used as a reasonable default optimizer that
adapts per-parameter learning rates without much tuning.

## Training epoch (`train_one_epoch`, TRAIN-2)

Standard supervised loop: move batch to device, zero gradients, forward,
compute loss, backward, step, then accumulate metrics.

```python
model.train()
running_loss, correct, total = 0.0, 0, 0

for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

average_loss = running_loss / len(loader)   # per-batch average
accuracy = correct / total                   # per-sample accuracy
```

The two denominators matter and are easy to swap by mistake: `running_loss`
accumulates one value **per batch**, so it's averaged over `len(loader)`
(number of batches); `correct` accumulates **per sample**, so it's divided
by `total` (number of samples), not the batch count.

## Validation epoch (`validate`, TRAIN-3)

Same accounting, but with `model.eval()` and the whole loop wrapped in
`torch.no_grad()` — no gradient computation, and no `zero_grad()` /
`backward()` / `step()` calls, since validation never updates weights.

## Checkpointing (TRAIN-4)

After each epoch, if `val_acc` improved on the best seen so far, the model
is saved:

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "num_classes": NUM_CLASSES,
    }, CHECKPOINT_PATH)
```

Saving is keyed strictly on **validation** accuracy, not training accuracy —
this is what keeps the checkpoint at `models/steel_cnn_best.pt` from being
overwritten by an epoch where the model has simply memorized more of the
training set without generalizing better.

## An actual run

Hyperparameters: `epochs=20`, `batch_size=32`, `lr=0.001`, `device=cuda`.
Splits: train=8,497 / val=1,822 / test=1,822. Model: `SteelCNN(params=113,285)`.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Checkpoint saved |
|---|---|---|---|---|---|
| 1 | 1.0246 | 0.565 | 0.9336 | 0.593 | ✅ |
| 2 | 0.9485 | 0.609 | 0.9085 | 0.616 | ✅ |
| 3 | 0.9203 | 0.623 | 0.8745 | 0.642 | ✅ |
| 4 | 0.8962 | 0.640 | 0.8183 | 0.680 | ✅ |
| 5 | 0.8673 | 0.651 | 0.8473 | 0.668 | |
| 6 | 0.8440 | 0.664 | 0.7815 | 0.679 | |
| 7 | 0.8164 | 0.678 | 0.7712 | 0.696 | ✅ |
| 8 | 0.8128 | 0.678 | 0.7459 | 0.716 | ✅ |
| 9 | 0.7948 | 0.691 | 0.7296 | 0.712 | |
| 10 | 0.7767 | 0.695 | 0.7349 | 0.688 | |
| 11 | 0.7658 | 0.702 | 0.7367 | 0.698 | |
| 12 | 0.7410 | 0.707 | 0.6604 | 0.748 | ✅ |
| 13 | 0.7248 | 0.721 | 0.6953 | 0.734 | |
| 14 | 0.7006 | 0.724 | 0.6390 | 0.761 | ✅ |
| 15 | 0.6950 | 0.732 | 0.8159 | 0.651 | |
| 16 | 0.6842 | 0.733 | 0.6458 | 0.742 | |
| 17 | 0.6704 | 0.741 | 0.6166 | 0.771 | ✅ |
| 18 | 0.6534 | 0.751 | 0.7217 | 0.702 | |
| 19 | 0.6495 | 0.748 | 0.5945 | 0.776 | ✅ |
| 20 | 0.6389 | 0.757 | 0.6323 | 0.773 | |

Total wall-clock time: **1,293.9s (~21.6 minutes)** on GPU, roughly 60–65s
per epoch after the first (epoch 1 included one-time dataloader/CUDA
warm-up, at ~111s). **Best validation accuracy: 77.6%**, reached at epoch 19
— that's the checkpoint saved to `models/steel_cnn_best.pt` and used by
`SteelPredictor` at inference time. See [Results](results.md) for discussion
of what this run's curves say about the model.

## Verification checkpoints

```bash
pytest tests/test_step4_style.py -v
pytest tests/test_step4_training.py -v   # TRAIN-1..3
python -m steel_defect.train --epochs 2  # sanity-check TRAIN-4 (checkpointing)
```
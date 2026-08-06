# Results

## Training Runs

| # | Epochs | LR | Batch | Train/Val/Test size | Best val acc | Notes |
|---|---|---|---|---|---|---|
| 1 | 20 | 0.001 | 32 | 8,497 / 1,822 / 1,822 | 77.6% (epoch 19) | Completed |
| 2 | 20 | 0.001 | 32 | 827 / 178 / 178 | 56.2% (epoch 20) | Small subset (274) |
| 3 | 50 | **0.1** | 32 | 827 / 178 / 178 | 20.8% | Aborted due to loss stuck at ~1.61 |
| 4 | 70 | 0.001 | 32 | 827 / 178 / 178 | 63.5% | Small Subset (274) |
| 5 | 50 | 0.01 | 32 | 2,435 / 522 / 523 | 70.9% | Medium Subset (1000) |
| 6 | 100 | 0.001 | 32 | 8,497 / 1,822 / 1,822 | **90.7% (epoch 94)** | Full Dataset, Completed, best run |

## Best run (#6)

- Epochs 100, batch 32, Adam `lr=0.001`, full dataset (8,497 train / 1,822 val / 1,822 test)
- **Best validation accuracy: 90.7%** at epoch 94 (`train_loss=0.310`, `train_acc=0.890`, `val_loss=0.304`)
- Final epoch (100): `train_acc=0.897`, `val_acc=0.878`
- Total training time: ~101 min (~60s/epoch)
- Loss/accuracy improve steadily but noisily epoch-to-epoch
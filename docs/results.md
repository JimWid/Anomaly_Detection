# Results

## Training run summary

From the run documented in [Training](training.md): 20 epochs, batch size 32,
Adam at `lr=0.001`, on 8,497 training / 1,822 validation images.

- **Best validation accuracy: 77.6%** (epoch 19)
- **Final-epoch train accuracy: 75.7%**, val accuracy: 77.3%
- **Total training time: ~21.6 minutes** on GPU (~60–65s/epoch after warm-up)

Train and validation loss both trend down over the run (1.02 → 0.64 train
loss, 0.93 → ~0.60–0.63 val loss), and validation accuracy tracks training
accuracy closely rather than diverging — a sign the model isn't badly
overfitting at 20 epochs. The narrow, converging train/val gap (train
accuracy never meaningfully exceeds validation accuracy) suggests there's
still headroom to train longer or increase model capacity before
overfitting becomes the limiting factor, rather than needing stronger
regularization right now.

## A rough patch

Epoch 15 is a visible outlier: validation loss jumps to 0.8159 (val acc
drops to 0.651) after epoch 14's 0.6390 / 0.761, before recovering by
epoch 17. Training loss/accuracy show no corresponding disruption at epoch
15, which points to this being validation-side instability for that
particular epoch's weights, rather than a training-data or pipeline issue.
This kind of single-epoch dip is common with plain Adam and no learning-rate
schedule; a scheduler (e.g. `ReduceLROnPlateau` or cosine annealing) is a
natural next experiment if this recurs across runs.

## What's not yet measured: held-out test accuracy

`create_splits()` reserves 1,822 images as a **test** set (15%), completely
separate from the 1,822 used for validation, but `train.py` never evaluates
on it — checkpointing and the numbers above are both driven purely by
validation accuracy. Since the same validation set is checked every epoch to
decide whether to save a checkpoint, validation accuracy is a somewhat
optimistic estimate of true generalization (the model selection process has
implicitly "seen" it, even though no gradient updates use it directly).

**The test set has not been scored against the final checkpoint.** Running
that evaluation — load `models/steel_cnn_best.pt` via `SteelPredictor`,
iterate `test_list` from `create_splits()`, and compute accuracy /
confusion matrix — would give the real unbiased number for this model and
is the natural next step before treating 77.6% as the model's true accuracy.

## Inference in production (Streamlit app)

The app's Grad-CAM + prediction flow has been exercised manually; a sample
of 109 real inference calls logged by `SteelPredictor.predict()`
(`final_project/logs/app.log`) shows:

| Metric | Value |
|---|---|
| Inferences logged | 109 |
| Average confidence | 0.52 |
| Confidence range | 0.23 – 1.00 |
| Average latency | 27.0 ms |
| Latency range | 2.7 – 100.8 ms |

Two things worth flagging from this sample rather than treating it as a
clean benchmark:

- **Latency spread** (2.7 ms to 100.8 ms) is wide for a fixed model and
  input size — almost certainly reflects a mix of cold CUDA kernel launches
  / Streamlit rerun overhead on some calls and steady-state GPU inference on
  others, not the model itself varying in cost.
- **Class distribution is heavily skewed toward `defect_3`** (64 of 109
  predictions), with `defect_2` and `defect_4` barely represented (2 and 3
  predictions respectively). This is more likely to reflect which images
  were browsed/tested manually in that session than a property of the
  model, but it's exactly the kind of pattern that's worth checking against
  a real confusion matrix on the test set — persistent over-prediction of
  one class would show up there clearly.

## Confidence as a signal

Average prediction confidence (0.52) is noticeably lower than the model's
77.6% validation accuracy, and the confidence range extends down to 0.23 —
close to the ~0.20 a uniform 5-class guess would produce. That gap between
"usually correct" (per validation accuracy) and "often not very confident"
(per softmax output) is typical of a network trained from scratch without
temperature calibration; softmax confidence out of an uncalibrated
classifier shouldn't be read as a literal probability of correctness. If the
app is meant to flag uncertain predictions for human review, calibrating
confidence (e.g. temperature scaling on the validation set) would make that
threshold much more meaningful than raw softmax output.
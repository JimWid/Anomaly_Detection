# Challenges

This page documents real bugs hit while implementing the 14 placeholders,
what broke, how it was diagnosed, and the fix — kept here as a debugging
log rather than a polished "everything worked first try" narrative.

## DATA-1: inverted directory check silently dropped every folder

**Symptom:** `build_file_list()` returned `[]` on a dataset with real images
in it — no exception, just zero results, which cascaded into `IndexError`
in dataset tests and `n_samples=0` errors from `train_test_split` in the
split tests, since both depended on a non-empty file list.

**Cause:**

```python
if folder.is_dir() or folder.name not in CLASS_NAMES:
    continue
```

For a valid class folder, `folder.is_dir()` is `True`, so the `or`
short-circuits to `True` and the folder gets skipped — every directory was
being treated as "not a match," including the correct ones. The condition
needed to skip only when the entry is **not** a directory, or its name
isn't a known class:

```python
if not folder.is_dir() or folder.name not in CLASS_NAMES:
    continue
```

**Lesson:** a silent empty-result bug (no traceback) is harder to spot than
a crash — tracing it back to DATA-1 required checking each downstream
failure's root dependency rather than debugging each test in isolation.

## DATA-1 (again): `list.append()` given two positional args

**Symptom:** `TypeError: list.append() takes exactly one argument (2 given)`,
surfacing only after the `is_dir()` fix above (the append line was never
reached before that, since every folder was being skipped).

**Cause:** `results.append(str(image_path), label)` — `.append()` takes a
single value; passing `str(image_path)` and `label` as two separate
arguments doesn't implicitly bundle them into a tuple. Needed an extra pair
of parentheses to make it one tuple argument:

```python
results.append((str(image_path), label))
```

## DATA-3: second split reused the full dataset, not the remainder

**Symptom:** none of the automated tests directly caught this one at first
glance — `create_splits` ran without error and returned three lists of
roughly the right sizes. The bug is a **correctness** issue (train/test
leakage), not a crash, and would only show up under
`test_no_overlap` (train/val/test path sets must be disjoint).

**Cause:** the naive first draft split the test set out correctly, but the
second `train_test_split` call was made on the original `file_list` again
instead of on the train+val remainder from the first split — meaning the
same samples could land in both the test set and the train/val sets, and
the `stratify` labels passed to the second call didn't match the subset
actually being split. Fixed by explicitly capturing and reusing the
train+val remainder:

```python
train_val_split, test_split = train_test_split(
    file_list, test_size=test_ratio, stratify=labels, random_state=seed)

train_val_labels = [label for _, label in train_val_split]
new_val_ratio = val_ratio / (train_ratio + val_ratio)

train_split, val_split = train_test_split(
    train_val_split, test_size=new_val_ratio,
    stratify=train_val_labels, random_state=seed)
```

**Lesson:** data-leakage bugs don't crash and don't always fail an
"approximate ratio" test — they need a dedicated overlap check
(`test_no_overlap`) to catch.

## MODEL-1/2: residual shortcuts with mismatched shapes

**Symptom:** runtime shape errors when adding the shortcut path back into
the main path (`residual_block_N(x) + x`).

**Cause:** an early draft tried adding the raw network input straight into
later blocks' outputs — but each `residual_block_N` both changes channel
count (via its conv) and halves spatial size (via `MaxPool2d(2)`), so a
raw, unprojected identity tensor from several blocks back never matches
shape. The fix was a dedicated `shortcut_N` (1×1 conv, `stride=2`, matching
output channels, + BatchNorm) for **each** block, added to that block's own
output — keeping every residual connection local rather than reaching back
to the original input. See [Architecture](model-architecture.md) for the
full corrected design. A separate, unrelated bug in the same file —
`nn.Sequential([...])` (a list) instead of `nn.Sequential(...)` (unpacked
positional args) — also had to be fixed, along with a missing
`super().__init__()` call, which is required before any submodules can be
registered on an `nn.Module` subclass.

## TRAIN-2/3: loss and accuracy divided by the wrong denominator

**Symptom:** no crash — `train_one_epoch` and `validate` both ran and
returned plausible-looking numbers, making this the easiest of these bugs
to miss without deliberately sanity-checking the math.

**Cause:**

```python
n = loader.batch_size
average_loss = running_loss / n
accuracy = correct / len(loader)
```

`running_loss` accumulates one value **per batch**, so dividing by
`loader.batch_size` (the size of a single batch, e.g. 32) rather than
`len(loader)` (the number of batches) gives a meaningless number.
Symmetrically, `correct` accumulates **per sample**, so dividing by
`len(loader)` (batch count) instead of `total` (sample count) is also
wrong. The fix swaps both denominators to what they should have been:

```python
average_loss = running_loss / len(loader)
accuracy = correct / total
```

The same bug was copy-pasted from `train_one_epoch` into `validate` and had
to be fixed in both places.

## INFER-1: loaded model never assigned to `self.model`

**Symptom:** `predict()` kept calling `load_model()` on every single
prediction (since `self.model` stayed `None`), rather than caching it once.

**Cause:** `load_model()` built a local `model` variable, loaded weights
into it, moved it to device, and set it to eval — but never ran
`self.model = model`, so the fully-loaded model was discarded at the end of
the function. Also missing: the explicit `FileNotFoundError` check on
`self.checkpoint_path` called out in the spec, which would otherwise let a
less-informative error surface from deep inside `torch.load`.

## INFER-2: predicted index/label/confidence never unwrapped from tensors

**Symptom:** `NameError` for `label`, `confidence_val`, and `predicted_idx`
at the point where they're used in the logging call and the return dict.

**Cause:** `confidence, predicted = probs.max(dim=1)` gives tensors, not
plain Python values — the draft computed `class_scores` correctly (per the
provided hint) but never took the extra step of converting `predicted` to
an int via `.item()`, mapping that int through `CLASS_NAMES` to get a
string label, and converting `confidence` to a float via `.item()`.

## Streamlit app: Grad-CAM assumed a `model.features` attribute

**Symptom:** `AttributeError` inside `generate_gradcam()`, silently caught
by its `except Exception` and masked as "Grad-CAM failed" in the logs —
producing a blank heatmap on every prediction instead of a real one.

**Cause:** the scaffolded helper was written against the plain-CNN
`self.features` design from the original homework spec
(`model.features.children()`), but the actual `SteelCNN` has no `features`
attribute — its conv blocks are `residual_block_1/2/3` plus
`shortcut_1/2/3`. Fixed by pointing the layer search at
`predictor.model.residual_block_3.children()` instead, so it walks the
correct block's `Conv2d` for hooking.

**Lesson:** the broad `except Exception` here is useful for keeping the app
from crashing on an unexpected image, but it also means an architecture
mismatch like this fails silently as "no heatmap" rather than a visible
error — worth logging the specific exception type/message prominently
(as it does) and checking those logs rather than assuming a blank heatmap
means the model has nothing to show.
"""
organize_dataset.py

Sorts a flat folder of steel-defect images into class subdirectories
(no_defect, defect_1, defect_2, defect_3, defect_4) based on a
train.csv file with columns: ImageId, ClassId, EncodedPixels.

Rules applied:
  - EncodedPixels is ignored entirely.
  - An image can appear on multiple rows (one per defect class it has).
    If an image has MORE THAN ONE distinct ClassId, it is SKIPPED
    (excluded from the output entirely) rather than assigned to one class.
  - Any image found in the source folder that does NOT appear in
    train.csv at all is treated as "no_defect".
  - Images are COPIED (not moved), so your original folder is untouched.

Usage:
    python organize_dataset.py \
        --images-dir /path/to/train_images \
        --csv /path/to/train.csv \
        --output-dir /path/to/data/steel_defect \
        [--move] [--dry-run]

Result layout:
    output_dir/
        no_defect/
        defect_1/
        defect_2/
        defect_3/
        defect_4/
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["no_defect", "defect_1", "defect_2", "defect_3", "defect_4"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def load_labels(csv_path: Path) -> dict[str, set[str]]:
    """Read train.csv and map ImageId -> set of ClassId strings seen for it."""
    labels: dict[str, set[str]] = defaultdict(set)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"].strip()
            class_id = row["ClassId"].strip()
            labels[image_id].add(class_id)
    return labels


def classify_images(images_dir: Path, labels: dict[str, set[str]]):
    """
    Walk images_dir and decide a target class folder for each image.

    Returns:
        assignments: list of (source_path, class_folder_name)
        skipped_multi: list of image ids skipped for having >1 class
    """
    assignments = []
    skipped_multi = []

    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        classes = labels.get(path.name)

        if not classes:
            # Not listed in train.csv at all -> no defect
            assignments.append((path, "no_defect"))
        elif len(classes) == 1:
            class_id = next(iter(classes))
            folder = f"defect_{class_id}"
            if folder not in CLASS_NAMES:
                print(f"  ! Unknown ClassId '{class_id}' for {path.name}, skipping")
                continue
            assignments.append((path, folder))
        else:
            # Multiple distinct defect classes for this image -> skip entirely
            skipped_multi.append(path.name)

    return assignments, skipped_multi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path,
                         help="Folder containing the flat list of training images")
    parser.add_argument("--csv", required=True, type=Path,
                         help="Path to train.csv (ImageId, ClassId, EncodedPixels)")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Destination root; class subfolders are created here")
    parser.add_argument("--move", action="store_true",
                         help="Move files instead of copying them")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would happen without touching any files")
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    labels = load_labels(args.csv)
    assignments, skipped_multi = classify_images(args.images_dir, labels)

    # Create output class folders
    if not args.dry_run:
        for name in CLASS_NAMES:
            (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    # Copy/move files
    counts = defaultdict(int)
    for src, folder in assignments:
        dst = args.output_dir / folder / src.name
        counts[folder] += 1
        if args.dry_run:
            continue
        if args.move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)

    # Report
    print("\n=== Summary ===")
    for name in CLASS_NAMES:
        print(f"{name:12s}: {counts[name]:5d} images")
    print(f"{'skipped (multi-defect)':12s}: {len(skipped_multi):5d} images")
    if skipped_multi:
        preview = ", ".join(skipped_multi[:10])
        more = "" if len(skipped_multi) <= 10 else f" (+{len(skipped_multi) - 10} more)"
        print(f"  e.g. {preview}{more}")

    total = sum(counts.values())
    print(f"\nTotal placed: {total} images -> {args.output_dir}")
    if args.dry_run:
        print("(dry run — no files were actually copied/moved)")


if __name__ == "__main__":
    main()
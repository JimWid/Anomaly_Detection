# Installation

## Prerequisites

- Python 3.10 or later
- Git
- UV
- ~2 GB disk space (for dataset and model weights)

## Environment Setup

```bash
uv sync
```

## Download the Dataset

1. Download the **severstal-steel-defect-detection.zip** file from [Google Drive](https://drive.google.com/file/d/1Fkk1hHzKE9VJvjh_LZEIrLBkIqRpNUZH/view?usp=sharing).

2. Extract the zip and place the folder inside the `data/` directory.

3. Run `orgnaized_dataset.py`:
```bash
python organized_dataset.py `
--images-dir path/to/data/train_images `
--csv path/to/csv/train.csv `
--output-dir path/to/output
```

## Train the Model

PatchCore builds a feature memory bank from the "good" images. This is a one-time step:

```bash
python -m final_project/steel_defect.train --epochs 20
```

This takes approximately 20 minutes depending on your hardware. The model checkpoint is saved to `final_project/models`.

## Verify Installation

```bash
# Launch the app
uv run streamlit run final_project/steel_defect/app.py
```

The Streamlit app should open at `http://localhost:8501`.

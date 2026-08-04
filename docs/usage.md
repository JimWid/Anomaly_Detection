# Usage Guide

## Running the Streamlit App

```bash
uv run streamlit run final_project/steel_defect/app.py
```

The app opens at `http://localhost:8501` and provides:

- **Image source selector** — choose which test category to inspect (good, bent, color, flip, scratch)
- **Acquire & Inspect** — simulates capturing a frame and running anomaly detection
- **Anomaly heatmap** — visual overlay showing where the model detects anomalies
- **Inspection history** — running tally of good vs. defective results

### Adjusting Settings

Use the sidebar controls to:

- **Anomaly threshold**: Increase to be more lenient (fewer false positives), decrease to catch more subtle defects
- **Heatmap opacity**: Controls how strongly the anomaly overlay is blended onto the original image

## Running Training

If you need to retrain the model (e.g., after changing image size or backbone):

```bash
python -m steel_defect.train
```

## Serving Documentation

To preview these docs locally:

```bash
mkdocs serve
```

Opens at `http://localhost:8000`. Pages auto-reload when you edit the Markdown files in `docs/`.

To build static HTML:

```bash
mkdocs build
```

Output goes to `site/` directory.

## Reading Logs

The application writes structured logs to `logs/app.log` and to the console. To follow logs in real time while the app runs:

```bash
tail -f logs/app.log
```

Example log output:

```
2025-01-15 14:32:01 | INFO     | anomaly_detection.acquisition | Frame acquired | id=3 | source=000.png | buffer=4/20
2025-01-15 14:32:01 | INFO     | anomaly_detection.inference   | Inference #4 | label=Defective | score=0.7231 | threshold=0.50 | time=45.2ms
```

### Filtering Logs

```bash
# Show only inference results
grep "Inference" logs/app.log

# Show only errors and warnings
grep -E "ERROR|WARNING" logs/app.log

# Show acquisition stats
grep "Frame acquired" logs/app.log
```

## Running Tests

```bash
pytest tests/ -v
```

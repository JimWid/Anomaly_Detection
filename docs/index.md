# Anomaly Detection

An anomaly-based defect detection system for industrial quality control, built as part of the **Building AI-Powered Defect Detection Systems for Industrial Quality Control** course.

## What This Project Does

This application inspects **Steel Surface Defect Classification Dataset**, and classifies them with 4 different types of defect: **defect_1, defect_2, defect_3, defect_4, or no_defect** using PyTorch for a **Convolutional Network architecture** with a **residuals approach**. Plus with a Streamlit app that produces heatmaps showing exactly where the model identifies anomalies.

## Project Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive inspection UI with image display and metrics |
| Backend | PyTorch CNN Architecture | Anomaly detection model with heatmap generation |
| Camera Simulation | Custom Python module | Mimics industrial camera acquisition from dataset images |
| Documentation | MkDocs + Material | This site — project docs and guides |
| CI/CD | GitHub Actions | Automated testing and linting on every push |
| Logging | Python logging | Structured logs for debugging and monitoring |

## Quick Start

```bash
# Clone and set up
git clone JimWid/Anomaly_Detection
cd anomaly_detection

# This version uses UV
uv sync

# Train the model (one-time, ~20 minutes)
python -m steel_defect.train --epochs 20

# Launch the app
streamlit run final_project/steel_defect/app.py
```

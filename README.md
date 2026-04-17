# EE655 Course Project

## Cricket Shot Detection and Video Similarity Analysis

This repository contains our `EE655` course project on cricket shot classification using deep learning and a Streamlit-based demonstration app. The final working system is based on the notebook:

- `fork-of-cricket-shot-detection (3).ipynb`

The current app and inference pipeline were updated to follow that notebook's logic, checkpoints, sampling strategy experiments, and result summaries.

## Project Objective

The goal of this project is to automatically identify the batting shot played in a cricket video clip and to compare two clips for similarity in shot mechanics.

The project supports classification of 10 cricket shot classes:

- `cover`
- `defense`
- `flick`
- `hook`
- `late_cut`
- `lofted`
- `pull`
- `square_cut`
- `straight`
- `sweep`

## Final Deliverables

The final project includes:

- a trained notebook-based deep learning pipeline
- saved PyTorch checkpoints from notebook experiments
- experiment result CSVs and plots
- a Streamlit application for demo and evaluation
- PDF and JSON report export from the app

## Final Project Files

Main files used in the final version:

- `fork-of-cricket-shot-detection (3).ipynb`
  Main source notebook for the final model design and experiments
- `app.py`
  Streamlit frontend for uploading videos, running inference, comparison, and downloading reports
- `cricket_notebook_model.py`
  Inference-side implementation aligned with the notebook checkpoints
- `results_current/`
  Extracted checkpoints, CSV summaries, confusion matrices, and training curves
- `results (1).zip`
  Original result bundle uploaded for this project
- `requirements.txt`
  Python dependencies

## What We Have Done

This project was completed in the following stages:

1. Data preparation
   Video clips were organized into 10 cricket shot classes.
2. Frame extraction and sampling
   Each video was converted into fixed-size frame sequences using multiple sampling strategies.
3. Model design
   We tested EfficientNetB0-based CNN, GRU, and LSTM architectures.
4. Phase 1 experiment
   We compared architectures using uniform sampling.
5. Phase 2 experiment
   We compared uniform, motion, and hybrid sampling strategies.
6. Phase 3 experiment
   We compared prediction aggregation methods such as single, majority, weighted, and temporal voting.
7. Application development
   We built a Streamlit app to demonstrate classification and comparison on uploaded videos.
8. Result integration
   The app now reads the latest notebook result bundle and uses the updated checkpoints directly.

## Methodology

### 1. Feature Backbone

The final system uses `EfficientNetB0` as the CNN feature extractor.

Why this was chosen:

- strong visual feature extraction
- lower computational cost than larger backbones
- suitable for video frame encoding

### 2. Sequence Modeling

We tested three model variants:

- `CNN Only`
  Frame-level features are averaged over time and passed to a classifier.
- `GRU`
  Sequential features are modeled using a bidirectional GRU.
- `LSTM`
  Sequential features are modeled using an LSTM.

### 3. Sampling Strategies

We evaluated:

- `uniform`
  Evenly spaced frames across the clip
- `motion`
  Frames selected from high-motion regions with local context
- `hybrid`
  Combination of motion-based, uniform, and random frame selection

### 4. Aggregation Strategies

We evaluated:

- `single`
  One direct clip prediction
- `majority`
  Majority voting across predictions
- `weighted`
  Motion-weighted probability aggregation
- `temporal`
  Sliding-window temporal averaging

## Experimental Results

### Phase 1: Architecture Comparison

Results from `results_current/results/phase1.csv`:

| Architecture | Validation Accuracy (%) | Test Accuracy (%) | Test Macro-F1 | Overfitting Gap (%) | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN Only | 79.51 | 79.02 | 0.7838 | 16.23 | 13 |
| GRU | 71.71 | 74.15 | 0.7370 | 27.87 | 15 |
| LSTM | 67.80 | 73.66 | 0.7250 | 30.27 | 13 |

Best architecture from notebook experiments:

- `CNN Only`

### Phase 2: Sampling Strategy Comparison

Results from `results_current/results/phase2_corrected.csv`:

| Sampling | Validation Accuracy (%) | Test Accuracy (%) | Test Macro-F1 | Overfitting Gap (%) | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 71.71 | 74.15 | 0.7370 | 27.87 | 15 |
| Hybrid | 62.93 | 62.44 | 0.6203 | 32.60 | 15 |
| Motion | 57.56 | 57.07 | 0.5725 | 34.38 | 15 |

Best sampling strategy from notebook experiments:

- `Uniform`

### Phase 3: Aggregation Strategy Comparison

Computed from `results_current/results/phase3.csv`:

| Method | Accuracy (%) | Average Confidence (%) | Average Entropy |
| --- | ---: | ---: | ---: |
| Majority | 80.00 | 100.00 | ~0.0000 |
| Single | 80.00 | 96.99 | 0.0848 |
| Weighted | 62.00 | 87.81 | 0.2824 |
| Temporal | 56.00 | 50.83 | 1.2607 |

Important note:

- the best notebook setup for model selection is `CNN Only + Uniform`
- majority voting is a useful aggregation result from phase 3, but it is not the architecture/sampling selection result

## Final Interpretation

Based on the notebook experiments:

- `CNN Only` achieved the best test accuracy among architectures
- `Uniform sampling` performed better than motion and hybrid sampling
- `Majority` and `Single` voting both reached `80%` accuracy in phase 3
- `CNN Only` also showed the smallest overfitting gap, so it is the most stable model among the tested options

## Efficiency and Practical Performance

### Model Efficiency

The final solution is efficient in the following ways:

- EfficientNetB0 is lighter than many larger CNN backbones
- notebook checkpoints are loaded directly without retraining
- Streamlit app performs inference on uploaded clips in a simple demo pipeline
- no separate backend server is required for the final demo

### Accuracy Efficiency

The final model gives a good balance of:

- test accuracy
- lower overfitting
- manageable inference cost

For this project, `CNN Only + Uniform` is the most efficient overall choice because it gave the best architecture accuracy while keeping the pipeline simpler than recurrent alternatives.

### Runtime Expectation

Practical timing depends on machine, CPU/GPU, and video length. For the current demo app:

- app startup: usually `5 to 15 seconds`
- single short clip inference: usually `3 to 15 seconds`
- two-clip comparison: usually `6 to 30 seconds`

If full retraining is attempted from the notebook:

- on GPU: typically `tens of minutes to a few hours`
- on CPU only: can take `several hours` or longer depending on dataset size and hardware

These are practical estimates for demo and course-report discussion, not guaranteed fixed timings.

## Current Application Features

The Streamlit app supports:

- single video shot recognition
- two-video comparison
- model selection from available checkpoints
- sampling strategy selection
- voting strategy selection
- confidence timeline visualization
- sampled frame preview
- notebook experiment summary tables
- downloadable PDF report
- downloadable JSON report
- recent session history in the sidebar

## How To Run The Project

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Make sure result files are present

The app expects:

- `results (1).zip`
  or
- extracted files under `results_current/`

The project is already configured to use the latest result bundle automatically.

### 4. Start the app

```powershell
streamlit run app.py
```

If `streamlit` is not recognized, use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
```

### 5. Open in browser

Use any of these:

- `http://localhost:8501`
- `http://127.0.0.1:8501`

## How To Use The App

1. Open the Streamlit app in the browser.
2. Choose model, sampling, and voting settings from the sidebar.
3. Upload one video for shot classification.
4. Upload two videos if you want shot similarity comparison.
5. Click `Run Analysis` or `Run Comparison`.
6. Review prediction, confidence, timeline, sampled frames, and notes.
7. Download the PDF or JSON report if needed.

## Dependencies

Key libraries used:

- `streamlit`
- `torch`
- `torchvision`
- `opencv-python-headless`
- `numpy`
- `pandas`
- `Pillow`
- `reportlab`

The current `requirements.txt` also includes some packages from older experiments, but the active Streamlit app mainly depends on the libraries listed above.

## Repository Notes

- the final app is based on the notebook `fork-of-cricket-shot-detection (3).ipynb`
- older unnecessary local experiment files were cleaned from the working project folder
- the inference code was updated to match the saved notebook checkpoints exactly
- the app now reads the extracted notebook result bundle from `results_current/`

## Limitations

- performance depends on the quality and duration of uploaded videos
- some visually similar shots are still confused with each other
- retraining is not part of the Streamlit app flow
- network/external URLs may be blocked by firewall settings on some systems, so `localhost` is the safest option

## Suggested Viva / Report Summary

This project demonstrates that deep learning can be used to classify cricket batting shots from video clips with useful accuracy for academic and demo purposes. Among the tested models, `CNN Only` with `Uniform` frame sampling gave the best and most stable overall result in our notebook experiments. The project was extended into a working Streamlit interface so that users can upload videos, run predictions, compare clips, and generate reports directly from the trained notebook outputs.

## Authors

Update this section with your actual group member names, roll numbers, and department details before final submission.

- `Student 1 - Name / Roll No.`
- `Student 2 - Name / Roll No.`
- `Student 3 - Name / Roll No.`


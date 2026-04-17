# Cricket Shot Detection

Cricket shot classification and video comparison project built for `EE655`.

The final implementation in this repository is based on the updated notebook:

- `fork-of-cricket-shot-detection (3).ipynb`

That notebook is the main source of truth for the final model pipeline, experiment setup, checkpoints, and result summaries used by the app.

## Overview

This project detects the batting shot played in a cricket video and can also compare two clips for similarity.

The system supports these shot classes:

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

## Final Project Basis

The final app and inference flow are aligned with `fork-of-cricket-shot-detection (3).ipynb`.

This includes:

- notebook-trained checkpoint usage
- frame sampling strategies from the notebook
- voting strategies exposed in the app
- experiment CSV summaries and result plots
- Streamlit demo for single-video analysis and two-video comparison

## Repository Structure

- `fork-of-cricket-shot-detection (3).ipynb`
  Final updated notebook used as the project reference implementation
- `app.py`
  Streamlit frontend for uploading videos, viewing predictions, and comparing clips
- `cricket_notebook_model.py`
  Inference code that mirrors the notebook logic for sampling, loading checkpoints, and prediction
- `results_current/`
  Extracted checkpoints, result CSV files, and generated notebook outputs used by the app
- `requirements.txt`
  Python dependencies needed to run the project

## Method Summary

The project evaluates multiple components from the notebook pipeline:

- architectures: `CNN Only`, `GRU`, `LSTM`
- sampling strategies: `uniform`, `motion`, `hybrid`
- voting strategies: `single`, `majority`, `weighted`

The frontend now presents these notebook-driven options directly instead of using a separate custom result flow.

## Features

- cricket shot prediction from a single uploaded video
- similarity comparison between two uploaded videos
- checkpoint selection from notebook outputs
- sampling and voting strategy selection
- probability breakdown and confidence summary
- sampled frame preview
- notebook experiment summary tables
- PDF and JSON report download
- recent session history in the sidebar

## How To Run

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Make sure notebook results are available

The app expects one of the following:

- extracted outputs under `results_current/`
- or a results archive that can be extracted by the app on startup

### 4. Start the Streamlit app

```powershell
streamlit run app.py
```

## GitHub Deployment

This repository is ready to be pushed to GitHub as a standard project repo.

Recommended flow:

```powershell
git add app.py README.md "fork-of-cricket-shot-detection (3).ipynb"
git commit -m "Align app and docs with final notebook pipeline"
git push origin HEAD
```

## Important Note

For this final version, `fork-of-cricket-shot-detection (3).ipynb` should be treated as the authoritative final code and experiment reference.

If there is any wording mismatch between the UI text and older notes, the notebook should be considered correct.
